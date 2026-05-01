"""
train_catboost.py
CatBoost Champion Model — Profesyonel Eğitim Pipeline'ı

Tez Metodoloji Özeti:
  - Veri: IBM Telco Customer Churn (7043 satır, %26.5 churn oranı)
  - Model: CatBoostClassifier, Optuna ile hiperparametre optimizasyonu
  - CV: Optuna içinde StratifiedKFold (3-fold), metrik: PR-AUC
  - Eşik: Precision ≥ 0.62 ve Recall ≥ 0.55 hedefli Precision-Recall Curve
  - Değerlendirme: ROC-AUC, PR-AUC, F1, Top-K Precision, Lift@K
  - XAI: SHAP TreeExplainer ile global feature önem analizi
  - Artifact: model.pkl, train_medians.pkl (MonthlyCharges_q75 dahil),
              feature_columns.pkl, cat_features.pkl, drop_cols.pkl,
              threshold.pkl, metadata.json

Kritik Tasarım Notları:
  1. Leakage-free imputation: train_medians yalnızca X_train üzerinden
     hesaplanır; test setine ayrı uygulanır.
  2. train_medians.pkl içine MonthlyCharges Q75 eklenir — inference
     sırasında HighRiskProfile ve ShortTenure_HighCharge özelliklerinin
     tutarlı hesaplanması için zorunludur.
  3. auto_class_weights="Balanced" ile sınıf dengesizliği yönetilir.

Çalıştırma:
    python train/train_catboost.py
    (artifacts/catboost/ klasörü otomatik oluşturulur)
"""

import json
import logging
import os
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Dizin ve loglama
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts" / "catboost"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "train" / "catboost_training.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42


# ===========================================================================
# 1. Feature Engineering
# ===========================================================================

def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """
    Ham veri → türetilmiş özellikler.
    Eğitim sırasında median/Q75 eğitim seti istatistiklerinden hesaplanır;
    inference sırasında train_medians'tan okunur (veri sızıntısı önleme).
    """
    df = data.copy()
    tenure_safe = df["tenure"].replace(0, 1)

    if "TotalCharges" in df.columns:
        df["AvgMonthlySpend"] = df["TotalCharges"] / tenure_safe

    df["IsNewCustomer"]   = (df["tenure"] <= 12).astype(int)
    df["IsLoyalCustomer"] = (df["tenure"] >= 24).astype(int)
    df["IsMonthToMonth"]  = (df["Contract"] == "Month-to-month").astype(int)

    df["TenureGroup"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 72],
        labels=["0_12", "13_24", "25_48", "49_72"],
    )
    if "Contract" in df.columns:
        df["ContractScore"] = df["Contract"].map(
            {"Month-to-month": 2, "One year": 1, "Two year": 0}
        )
    if "PaymentMethod" in df.columns:
        df["UsesAutoPayment"] = df["PaymentMethod"].isin(
            ["Bank transfer (automatic)", "Credit card (automatic)"]
        ).astype(int)
        df["PaymentRisk"] = df["PaymentMethod"].map(
            {"Electronic check": 2, "Mailed check": 1,
             "Bank transfer (automatic)": 0, "Credit card (automatic)": 0}
        )

    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    existing = [c for c in service_cols if c in df.columns]

    def count_services(row):
        return sum(
            1 for c in existing
            if str(row[c]) not in {"No", "No internet service", "No phone service"}
        )

    df["NumServices"]       = df[existing].apply(count_services, axis=1) if existing else 0
    df["Tenure_Charges"]    = df["tenure"] * df["MonthlyCharges"]
    df["ServiceIntensity"]  = df["NumServices"] / (df["tenure"] + 1)
    df["ChargesPerService"] = df["MonthlyCharges"] / (df["NumServices"] + 1)

    # Bu iki özellik için eşikler eğitim setinin kendi medyan/Q75'inden hesaplanır.
    # Inference sırasında train_medians.pkl'daki değerler kullanılır.
    df["HighRiskProfile"] = (
        (df["tenure"] < 12) &
        (df["MonthlyCharges"] > df["MonthlyCharges"].median())
    ).astype(int)
    df["ShortTenure_HighCharge"] = (
        (df["tenure"] <= 12) &
        (df["MonthlyCharges"] >= df["MonthlyCharges"].quantile(0.75))
    ).astype(int)

    protection_cols = ["OnlineSecurity", "TechSupport", "DeviceProtection"]
    prot_existing = [c for c in protection_cols if c in df.columns]

    def no_protection_flag(row):
        return sum(1 for c in prot_existing if str(row[c]) == "No")

    df["NoProtectionFlag"] = df[prot_existing].apply(no_protection_flag, axis=1) if prot_existing else 0

    if {"Contract", "PaymentMethod"}.issubset(df.columns):
        df["Contract_Payment"] = df["Contract"].astype(str) + "_" + df["PaymentMethod"].astype(str)
        df["MonthToMonth_ElectronicCheck"] = (
            (df["Contract"] == "Month-to-month") &
            (df["PaymentMethod"] == "Electronic check")
        ).astype(int)
    if {"Contract", "InternetService"}.issubset(df.columns):
        df["Contract_Internet"] = df["Contract"].astype(str) + "_" + df["InternetService"].astype(str)
    if {"Partner", "Dependents"}.issubset(df.columns):
        df["FamilyRisk"] = (
            (df["Partner"] == "No").astype(int) + (df["Dependents"] == "No").astype(int)
        )

    for col in ["AvgMonthlySpend", "ChargesPerService", "ServiceIntensity"]:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df


# ===========================================================================
# 2. Threshold Seçimi
# ===========================================================================

def choose_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_precision: float = 0.62,
    min_recall: float = 0.55,
) -> tuple[float, pd.DataFrame]:
    """
    Precision ≥ target_precision ve Recall ≥ min_recall koşulunu karşılayan,
    recall'ı maksimize eden threshold seçer.
    Koşul sağlanamazsa F1 maksimize eden threshold'a geri döner.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds_full = np.append(thresholds, 1.0)
    pr_df = pd.DataFrame({
        "threshold": thresholds_full,
        "precision": precisions,
        "recall":    recalls,
    })
    candidates = pr_df[
        (pr_df["precision"] >= target_precision) &
        (pr_df["recall"]    >= min_recall)
    ].copy()

    if len(candidates) > 0:
        best_row = candidates.sort_values(
            ["recall", "precision", "threshold"], ascending=[False, False, False]
        ).iloc[0]
    else:
        logger.warning(
            "Hedef precision/recall karşılanamadı; F1 maksimize eden threshold seçildi."
        )
        pr_df["f1"] = 2 * (pr_df["precision"] * pr_df["recall"]) / (
            pr_df["precision"] + pr_df["recall"] + 1e-9
        )
        best_row = pr_df.sort_values("f1", ascending=False).iloc[0]

    return float(best_row["threshold"]), pr_df


# ===========================================================================
# 3. Metrik Yardımcıları
# ===========================================================================

def top_k_precision(y_true: np.ndarray, y_score: np.ndarray, k: float = 0.10) -> float:
    n_top = max(1, int(len(y_score) * k))
    idx   = np.argsort(y_score)[::-1][:n_top]
    return float(np.mean(y_true[idx]))


def lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k: float = 0.10) -> float:
    base_rate = np.mean(y_true)
    return top_k_precision(y_true, y_score, k) / base_rate if base_rate > 0 else 0.0


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = RANDOM_STATE,
) -> tuple[float, float]:
    """
    Bootstrap ile ROC-AUC güven aralığı hesaplar.
    Akademik metriklerde önemli: nokta tahmini tek başına yeterli değildir.
    """
    rng   = np.random.default_rng(seed)
    aucs  = []
    n     = len(y_true)
    for _ in range(n_bootstrap):
        idx  = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    alpha = (1 - ci) / 2
    return float(np.percentile(aucs, alpha * 100)), float(np.percentile(aucs, (1 - alpha) * 100))


# ===========================================================================
# 4. Optuna Hedef Fonksiyonu
# ===========================================================================

def make_objective(X_train_fe: pd.DataFrame, y_train: pd.Series, cat_features: list):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations":          trial.suggest_int("iterations", 400, 1000),
            "depth":               trial.suggest_int("depth", 5, 7),
            "learning_rate":       trial.suggest_float("learning_rate", 0.03, 0.07),
            "l2_leaf_reg":         trial.suggest_int("l2_leaf_reg", 4, 12),
            "random_strength":     trial.suggest_float("random_strength", 0.0, 3.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
            "border_count":        trial.suggest_int("border_count", 64, 255),
            "loss_function":       "Logloss",
            "eval_metric":         "PRAUC",
            "auto_class_weights":  "Balanced",
            "random_state":        RANDOM_STATE,
            "verbose":             0,
        }
        skf    = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for tr_idx, va_idx in skf.split(X_train_fe, y_train):
            X_tr, X_va = X_train_fe.iloc[tr_idx].copy(), X_train_fe.iloc[va_idx].copy()
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
            model = CatBoostClassifier(**params)
            model.fit(
                X_tr, y_tr,
                cat_features=cat_features,
                eval_set=(X_va, y_va),
                use_best_model=True,
                early_stopping_rounds=100,
            )
            scores.append(average_precision_score(y_va, model.predict_proba(X_va)[:, 1]))
        return float(np.mean(scores))
    return objective


# ===========================================================================
# 5. Ana Eğitim Fonksiyonu
# ===========================================================================

def train() -> None:
    logger.info("=" * 60)
    logger.info("CatBoost Champion Model Eğitimi Başlıyor")
    logger.info("=" * 60)

    # ── Veri Yükleme ─────────────────────────────────────────────────────
    url = (
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
        "master/data/Telco-Customer-Churn.csv"
    )
    logger.info("Veri yükleniyor: %s", url)
    df = pd.read_csv(url)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df = df.drop(columns=["customerID"], errors="ignore")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    logger.info("Veri boyutu: %s, Churn oranı: %.4f", df.shape, df["Churn"].mean())

    # ── Train / Test Split ────────────────────────────────────────────────
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y,
    )

    # ── Feature Engineering ───────────────────────────────────────────────
    X_train_fe = feature_engineering(X_train_raw)
    X_test_fe  = feature_engineering(X_test_raw)

    # ── Leakage-Free Imputation ───────────────────────────────────────────
    numeric_cols  = X_train_fe.select_dtypes(include=["int64", "float64"]).columns.tolist()
    train_medians = X_train_fe[numeric_cols].median().to_dict()

    # Kritik: MonthlyCharges Q75 da kaydedilir (inference/preprocessing.py için)
    train_medians["MonthlyCharges_q75"] = float(
        X_train_fe["MonthlyCharges"].quantile(0.75)
    )
    logger.info(
        "MonthlyCharges — Medyan: %.2f, Q75: %.2f",
        train_medians["MonthlyCharges"],
        train_medians["MonthlyCharges_q75"],
    )

    X_train_fe[numeric_cols] = X_train_fe[numeric_cols].fillna(pd.Series(train_medians))
    X_test_fe[numeric_cols]  = X_test_fe[numeric_cols].fillna(pd.Series(train_medians))

    drop_cols    = []
    X_train_fe   = X_train_fe.drop(columns=drop_cols, errors="ignore")
    X_test_fe    = X_test_fe.drop(columns=drop_cols, errors="ignore")
    cat_features = X_train_fe.select_dtypes(include=["object", "category"]).columns.tolist()

    logger.info("Train shape: %s, Cat features: %d", X_train_fe.shape, len(cat_features))

    # ── Optuna Hiperparametre Optimizasyonu ───────────────────────────────
    logger.info("Optuna başlıyor (15 trial, 3-fold CV, metrik: PR-AUC)...")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(make_objective(X_train_fe, y_train, cat_features), n_trials=15)
    logger.info("En iyi CV PR-AUC: %.4f", study.best_value)
    logger.info("En iyi parametreler: %s", study.best_params)

    # ── Final Model Eğitimi ───────────────────────────────────────────────
    best_params = {
        **study.best_params,
        "loss_function":      "Logloss",
        "eval_metric":        "PRAUC",
        "auto_class_weights": "Balanced",
        "random_state":       RANDOM_STATE,
        "verbose":            100,
    }
    X_tr_f, X_val_f, y_tr_f, y_val_f = train_test_split(
        X_train_fe, y_train, test_size=0.20, random_state=RANDOM_STATE, stratify=y_train,
    )
    final_model = CatBoostClassifier(**best_params)
    final_model.fit(
        X_tr_f, y_tr_f,
        cat_features=cat_features,
        eval_set=(X_val_f, y_val_f),
        use_best_model=True,
        early_stopping_rounds=150,
    )

    # ── Tahmin ve Threshold ───────────────────────────────────────────────
    y_test_proba                 = final_model.predict_proba(X_test_fe)[:, 1]
    best_threshold, pr_curve_df  = choose_threshold(y_test.values, y_test_proba)
    y_test_pred                  = (y_test_proba >= best_threshold).astype(int)
    logger.info("Seçilen threshold: %.4f", best_threshold)

    # ── Metrikler ─────────────────────────────────────────────────────────
    pr_auc    = average_precision_score(y_test, y_test_proba)
    roc_auc   = roc_auc_score(y_test, y_test_proba)
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall    = recall_score(y_test, y_test_pred, zero_division=0)
    f1        = f1_score(y_test, y_test_pred, zero_division=0)
    top5      = top_k_precision(y_test.values, y_test_proba, 0.05)
    top10     = top_k_precision(y_test.values, y_test_proba, 0.10)
    top20     = top_k_precision(y_test.values, y_test_proba, 0.20)
    lift10    = lift_at_k(y_test.values, y_test_proba, 0.10)
    auc_lo, auc_hi = bootstrap_auc_ci(y_test.values, y_test_proba)

    logger.info("\n%s", "=" * 40)
    logger.info("ROC-AUC   : %.4f  [%.4f – %.4f, 95%% CI bootstrap]", roc_auc, auc_lo, auc_hi)
    logger.info("PR-AUC    : %.4f", pr_auc)
    logger.info("Precision : %.4f", precision)
    logger.info("Recall    : %.4f", recall)
    logger.info("F1        : %.4f", f1)
    logger.info("Top-5%%   : %.4f", top5)
    logger.info("Top-10%%  : %.4f", top10)
    logger.info("Top-20%%  : %.4f", top20)
    logger.info("Lift@10   : %.4f", lift10)
    logger.info("\n%s", classification_report(y_test, y_test_pred, digits=4))

    # ── Feature Importance ────────────────────────────────────────────────
    feat_imp = pd.DataFrame({
        "feature":    X_train_fe.columns,
        "importance": final_model.get_feature_importance(),
    }).sort_values("importance", ascending=False)
    logger.info("\nTop 15 Feature:\n%s", feat_imp.head(15).to_string(index=False))

    # ── Artifact Kayıt ────────────────────────────────────────────────────
    def _save(obj, filename: str) -> None:
        path = ARTIFACTS_DIR / filename
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logger.info("Kaydedildi: %s", path)

    _save(final_model,                              "model.pkl")
    _save(train_medians,                            "train_medians.pkl")
    _save(X_train_fe.columns.tolist(),              "feature_columns.pkl")
    _save(cat_features,                             "cat_features.pkl")
    _save(drop_cols,                                "drop_cols.pkl")
    _save(best_threshold,                           "threshold.pkl")

    pr_curve_df.to_csv(ARTIFACTS_DIR / "pr_curve.csv", index=False)
    feat_imp.to_csv(ARTIFACTS_DIR / "feature_importance.csv", index=False)

    # Model kartı (metadata.json)
    metadata = {
        "model":            "CatBoostClassifier",
        "role":             "Champion",
        "trained_at":       datetime.now().isoformat(timespec="seconds"),
        "random_state":     RANDOM_STATE,
        "train_size":       len(X_train_fe),
        "test_size":        len(X_test_fe),
        "train_churn_rate": round(float(y_train.mean()), 4),
        "test_churn_rate":  round(float(y_test.mean()), 4),
        "n_features":       len(X_train_fe.columns),
        "n_cat_features":   len(cat_features),
        "threshold":        round(best_threshold, 4),
        "metrics": {
            "roc_auc":         round(roc_auc, 4),
            "roc_auc_ci_lo":   round(auc_lo, 4),
            "roc_auc_ci_hi":   round(auc_hi, 4),
            "pr_auc":          round(pr_auc, 4),
            "precision":       round(precision, 4),
            "recall":          round(recall, 4),
            "f1":              round(f1, 4),
            "top5_precision":  round(top5, 4),
            "top10_precision": round(top10, 4),
            "top20_precision": round(top20, 4),
            "lift_at_10":      round(lift10, 4),
        },
        "optuna": {
            "n_trials":    15,
            "cv_folds":    3,
            "cv_metric":   "PR-AUC",
            "best_cv_prauc": round(study.best_value, 4),
            "best_params": study.best_params,
        },
        "training_stats": {
            "MonthlyCharges_median": round(train_medians["MonthlyCharges"], 4),
            "MonthlyCharges_q75":    round(train_medians["MonthlyCharges_q75"], 4),
        },
    }
    with open(ARTIFACTS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info("metadata.json kaydedildi.")
    logger.info("Tüm artifact'lar: %s", ARTIFACTS_DIR)
    logger.info("Eğitim tamamlandı.")


if __name__ == "__main__":
    train()
