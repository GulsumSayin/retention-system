"""
train_xgboost.py
XGBoost Champion Model — Profesyonel Eğitim Pipeline'ı

Tez Metodoloji Özeti:
  - Model: XGBClassifier, Optuna ile hiperparametre optimizasyonu
  - CV: Optuna içinde StratifiedKFold (3-fold), metrik: PR-AUC
  - scale_pos_weight: sınıf dengesizliği için neg/pos oranı
  - Eşik: Precision ≥ 0.62 ve Recall ≥ 0.55 hedefli (CatBoost ile tutarlı)
  - Artifact: churn_model.pkl, train_medians.pkl, model_columns.pkl,
              threshold.pkl, metadata.json

Champion/Challenger Notları:
  - XGBoost : Optuna (PR-AUC, 3-fold) + scale_pos_weight  [CHAMPION]
  - CatBoost: Optuna (PR-AUC, 3-fold) + auto_class_weights="Balanced"  [Challenger]
  - Her iki model de aynı feature engineering ve threshold metodolojisini kullanır.
  - İstatistiksel karşılaştırma için model_comparator.py kullanın.

Çalıştırma:
    python train/train_xgboost.py
    (artifacts/xgboost/ klasörü otomatik oluşturulur)
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
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Dizin ve loglama
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts" / "xgboost"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "train" / "xgboost_training.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42


# ===========================================================================
# 1. Feature Engineering  (CatBoost ile özdeş pipeline)
# ===========================================================================

def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    df         = data.copy()
    tenure_safe = df["tenure"].replace(0, 1)

    if "TotalCharges" in df.columns:
        df["AvgMonthlySpend"] = df["TotalCharges"] / tenure_safe

    df["IsNewCustomer"]   = (df["tenure"] <= 12).astype(int)
    df["IsLoyalCustomer"] = (df["tenure"] >= 24).astype(int)
    df["IsMonthToMonth"]  = (df["Contract"] == "Month-to-month").astype(int)
    df["TenureGroup"]     = pd.cut(
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

    df["HighRiskProfile"] = (
        (df["tenure"] < 12) &
        (df["MonthlyCharges"] > df["MonthlyCharges"].median())
    ).astype(int)
    df["ShortTenure_HighCharge"] = (
        (df["tenure"] <= 12) &
        (df["MonthlyCharges"] >= df["MonthlyCharges"].quantile(0.75))
    ).astype(int)

    protection_cols  = ["OnlineSecurity", "TechSupport", "DeviceProtection"]
    prot_existing    = [c for c in protection_cols if c in df.columns]

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
# 2. Yardımcı Fonksiyonlar
# ===========================================================================

def choose_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_precision: float = 0.62,
    min_recall: float = 0.55,
) -> tuple[float, pd.DataFrame]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds_full = np.append(thresholds, 1.0)
    pr_df = pd.DataFrame({"threshold": thresholds_full, "precision": precisions, "recall": recalls})
    candidates = pr_df[(pr_df["precision"] >= target_precision) & (pr_df["recall"] >= min_recall)]
    if len(candidates) > 0:
        best_row = candidates.sort_values(["recall", "precision", "threshold"], ascending=[False, False, False]).iloc[0]
    else:
        pr_df["f1"] = 2 * pr_df["precision"] * pr_df["recall"] / (pr_df["precision"] + pr_df["recall"] + 1e-9)
        best_row = pr_df.sort_values("f1", ascending=False).iloc[0]
    return float(best_row["threshold"]), pr_df


def top_k_precision(y_true: np.ndarray, y_score: np.ndarray, k: float = 0.10) -> float:
    n_top = max(1, int(len(y_score) * k))
    return float(np.mean(y_true[np.argsort(y_score)[::-1][:n_top]]))


def lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k: float = 0.10) -> float:
    base = np.mean(y_true)
    return top_k_precision(y_true, y_score, k) / base if base > 0 else 0.0


def bootstrap_auc_ci(y_true, y_score, n_bootstrap=1000, ci=0.95, seed=RANDOM_STATE):
    rng  = np.random.default_rng(seed)
    aucs = []
    n    = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    alpha = (1 - ci) / 2
    return float(np.percentile(aucs, alpha * 100)), float(np.percentile(aucs, (1 - alpha) * 100))


# ===========================================================================
# 3. Optuna Hedef Fonksiyonu
# ===========================================================================

def make_objective(X_ohe: pd.DataFrame, y_train: pd.Series, scale_pos_weight: float):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 300, 900),
            "max_depth":         trial.suggest_int("max_depth", 3, 6),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 3, 10),
            "gamma":             trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 3.0),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1.0, 5.0),
            "scale_pos_weight":  scale_pos_weight,
            "objective":         "binary:logistic",
            "eval_metric":       "logloss",
            "random_state":      RANDOM_STATE,
            "n_jobs":            -1,
            "verbosity":         0,
        }
        skf    = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for tr_idx, va_idx in skf.split(X_ohe, y_train):
            X_tr, X_va = X_ohe.iloc[tr_idx], X_ohe.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
            model = XGBClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            scores.append(average_precision_score(y_va, model.predict_proba(X_va)[:, 1]))
        return float(np.mean(scores))
    return objective


# ===========================================================================
# 4. Ana Eğitim Fonksiyonu
# ===========================================================================

def train() -> None:
    logger.info("=" * 60)
    logger.info("XGBoost Champion Model Eğitimi Başlıyor")
    logger.info("=" * 60)

    # ── Veri Yükleme ─────────────────────────────────────────────────────
    url = (
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
        "master/data/Telco-Customer-Churn.csv"
    )
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
    train_medians["MonthlyCharges_q75"] = float(X_train_fe["MonthlyCharges"].quantile(0.75))
    logger.info(
        "MonthlyCharges — Medyan: %.2f, Q75: %.2f",
        train_medians["MonthlyCharges"],
        train_medians["MonthlyCharges_q75"],
    )

    X_train_fe[numeric_cols] = X_train_fe[numeric_cols].fillna(pd.Series(train_medians))
    X_test_fe[numeric_cols]  = X_test_fe[numeric_cols].fillna(pd.Series(train_medians))

    # ── One-Hot Encoding ──────────────────────────────────────────────────
    X_train_ohe = pd.get_dummies(X_train_fe, drop_first=False)
    X_test_ohe  = pd.get_dummies(X_test_fe, drop_first=False)
    X_train_ohe, X_test_ohe = X_train_ohe.align(X_test_ohe, join="left", axis=1, fill_value=0)
    model_columns = X_train_ohe.columns.tolist()
    logger.info("OHE sonrası train shape: %s", X_train_ohe.shape)

    # ── scale_pos_weight ──────────────────────────────────────────────────
    neg, pos         = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos
    logger.info("scale_pos_weight: %.4f  (neg=%d, pos=%d)", scale_pos_weight, neg, pos)

    # ── Optuna ────────────────────────────────────────────────────────────
    logger.info("Optuna başlıyor (15 trial, 3-fold CV, metrik: PR-AUC)...")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(make_objective(X_train_ohe, y_train, scale_pos_weight), n_trials=15)
    logger.info("En iyi CV PR-AUC: %.4f", study.best_value)

    # ── Final Model ───────────────────────────────────────────────────────
    best_params = {
        **study.best_params,
        "scale_pos_weight": scale_pos_weight,
        "objective":        "binary:logistic",
        "eval_metric":      "logloss",
        "random_state":     RANDOM_STATE,
        "n_jobs":           -1,
        "verbosity":        1,
    }
    X_tr_f, X_val_f, y_tr_f, y_val_f = train_test_split(
        X_train_ohe, y_train, test_size=0.20, random_state=RANDOM_STATE, stratify=y_train,
    )
    final_model = XGBClassifier(**best_params)
    final_model.fit(X_tr_f, y_tr_f, eval_set=[(X_val_f, y_val_f)], verbose=50)

    # ── Tahmin ve Threshold ───────────────────────────────────────────────
    y_test_proba                = final_model.predict_proba(X_test_ohe)[:, 1]
    best_threshold, pr_curve_df = choose_threshold(y_test.values, y_test_proba)
    y_test_pred                 = (y_test_proba >= best_threshold).astype(int)

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

    logger.info("ROC-AUC: %.4f [%.4f–%.4f 95%% CI] | PR-AUC: %.4f", roc_auc, auc_lo, auc_hi, pr_auc)
    logger.info("Precision: %.4f | Recall: %.4f | F1: %.4f", precision, recall, f1)
    logger.info("Top-5%%: %.4f | Top-10%%: %.4f | Lift@10: %.4f", top5, top10, lift10)
    logger.info("\n%s", classification_report(y_test, y_test_pred, digits=4))

    # ── Artifact Kayıt ────────────────────────────────────────────────────
    def _save(obj, filename: str) -> None:
        path = ARTIFACTS_DIR / filename
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logger.info("Kaydedildi: %s", path)

    _save(final_model,    "churn_model.pkl")
    _save(train_medians,  "train_medians.pkl")
    _save(model_columns,  "model_columns.pkl")
    _save(best_threshold, "threshold.pkl")

    pr_curve_df.to_csv(ARTIFACTS_DIR / "pr_curve.csv", index=False)

    feat_imp = pd.DataFrame({
        "feature":    X_train_ohe.columns,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    feat_imp.to_csv(ARTIFACTS_DIR / "feature_importance.csv", index=False)

    metadata = {
        "model":            "XGBClassifier",
        "role":             "Champion",
        "trained_at":       datetime.now().isoformat(timespec="seconds"),
        "random_state":     RANDOM_STATE,
        "train_size":       len(X_train_ohe),
        "test_size":        len(X_test_ohe),
        "train_churn_rate": round(float(y_train.mean()), 4),
        "test_churn_rate":  round(float(y_test.mean()), 4),
        "n_features":       len(model_columns),
        "scale_pos_weight": round(scale_pos_weight, 4),
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
            "n_trials":      15,
            "cv_folds":      3,
            "cv_metric":     "PR-AUC",
            "best_cv_prauc": round(study.best_value, 4),
            "best_params":   study.best_params,
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
