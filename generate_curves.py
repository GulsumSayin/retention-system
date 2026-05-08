"""
generate_curves.py
------------------
Egitilmis CatBoost ve XGBoost modellerini yukler, test verisi uzerinde
tahmin uretir ve ROC ile Hassasiyet-Duyarlilik egri grafiklerini kaydeder.

Kullanim:
    python generate_curves.py                   # varsayilan yolu dener
    python generate_curves.py veri.csv          # komut satiri argumani
    python generate_curves.py --path veri.csv   # --path bayragi

Veri Gereksinimi:
    IBM Telco formatinda CSV, 'Churn' sutunu icermeli.
    Ornek konum: data/WA_Fn-UseC_-Telco-Customer-Churn.csv
"""

import os
import sys
import pickle
import argparse

# --------------------------------------------------------------------- #
# Calisma dizinini retention_system koku olarak ayarla                  #
# --------------------------------------------------------------------- #
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# src/ klasorunu import yoluna ekle
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # GUI olmayan ortamlar icin
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
)

from preprocessing import feature_engineering, prepare_cat_input

# --------------------------------------------------------------------- #
# Sabitler                                                               #
# --------------------------------------------------------------------- #
DEFAULT_CSV_PATHS = [
    "data/telco_churn.csv",
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "data/Telco-Customer-Churn.csv",
    "data/telco.csv",
]

OUTPUT_DIR = "outputs"

CAT_COLOR = "#1f77b4"   # mavi
XGB_COLOR = "#ff7f0e"   # turuncu


# --------------------------------------------------------------------- #
# Yardimci: model dosyalarini yukle                                      #
# --------------------------------------------------------------------- #
def load_artifacts():
    """Tum model artifact dosyalarini pickle ile yukler."""
    def _load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    print("[1/5] Artifact dosyalari yukleniyor...")

    cat_model      = _load("artifacts/catboost/model.pkl")
    cat_threshold  = _load("artifacts/catboost/threshold.pkl")
    cat_medians    = _load("artifacts/catboost/train_medians.pkl")
    cat_features   = _load("artifacts/catboost/feature_columns.pkl")
    cat_drop       = _load("artifacts/catboost/drop_cols.pkl")

    xgb_model      = _load("artifacts/xgboost/churn_model.pkl")
    xgb_threshold  = _load("artifacts/xgboost/threshold.pkl")
    xgb_columns    = _load("artifacts/xgboost/model_columns.pkl")
    xgb_medians    = _load("artifacts/xgboost/train_medians.pkl")

    # Threshold skaler olabilir (ndarray veya float)
    cat_threshold = float(np.atleast_1d(cat_threshold)[0])
    xgb_threshold = float(np.atleast_1d(xgb_threshold)[0])

    print(f"    CatBoost esigi : {cat_threshold:.4f}")
    print(f"    XGBoost esigi  : {xgb_threshold:.4f}")

    return (
        cat_model, cat_threshold, cat_medians, cat_features, cat_drop,
        xgb_model, xgb_threshold, xgb_columns, xgb_medians,
    )


# --------------------------------------------------------------------- #
# Yardimci: CSV bul / sor                                               #
# --------------------------------------------------------------------- #
def resolve_csv_path(cli_path=None):
    """Kullanicidan veya varsayilan yollardan CSV dosyasini bulur."""
    if cli_path:
        path = os.path.abspath(cli_path)
        if not os.path.isfile(path):
            sys.exit(f"HATA: Belirtilen dosya bulunamadi: {path}")
        return path

    # Varsayilan yollar
    for p in DEFAULT_CSV_PATHS:
        full = os.path.join(SCRIPT_DIR, p)
        if os.path.isfile(full):
            print(f"[CSV] Varsayilan veri seti bulundu: {full}")
            return full

    # Kullaniciya sor
    print("\nVeri seti bulunamadi. Lutfen CSV yolunu girin")
    print("(IBM Telco formati, 'Churn' sutunu gerekli):")
    user_path = input("CSV yolu: ").strip().strip('"').strip("'")
    if not user_path:
        sys.exit("HATA: Yol girilmedi, cikiliyor.")
    full = os.path.abspath(user_path)
    if not os.path.isfile(full):
        sys.exit(f"HATA: Dosya bulunamadi: {full}")
    return full


# --------------------------------------------------------------------- #
# Veri yukle ve test bolumunu ayir                                       #
# --------------------------------------------------------------------- #
def load_and_split(csv_path):
    """Ham CSV'yi yukler ve egitimle ayni 80/20 bolumlemesini yapar."""
    print(f"[2/5] Veri yukleniyor: {csv_path}")
    df = pd.read_csv(csv_path)

    if "Churn" not in df.columns:
        sys.exit(
            "HATA: CSV dosyasinda 'Churn' sutunu bulunamadi.\n"
            "IBM Telco formatinda bir dosya kullanin."
        )

    # Churn sutununu ikili sayiya donustur
    df["Churn"] = df["Churn"].map(
        {"Yes": 1, "No": 0, 1: 1, 0: 0, True: 1, False: 0}
    ).astype(int)

    print(f"    Toplam kayit : {len(df)}")
    print(f"    Churn orani  : {df['Churn'].mean():.3f}")

    # Egitimle ayni train/test bolumlemesi
    _, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["Churn"],
        random_state=42,
    )

    y_true = test_df["Churn"].values
    print(f"    Test kayit   : {len(test_df)} | Test churn orani: {y_true.mean():.3f}")

    return test_df, y_true


# --------------------------------------------------------------------- #
# Onisleme ve tahmin                                                     #
# --------------------------------------------------------------------- #
def get_predictions(
    test_df,
    cat_model, cat_medians, cat_features, cat_drop,
    xgb_model, xgb_columns, xgb_medians,
):
    """Her iki model icin olasilik tahmini uretir."""
    print("[3/5] On isleme ve tahminler yapiliyor...")

    # -- CatBoost --
    X_cat = prepare_cat_input(
        test_df,
        train_medians=cat_medians,
        feature_columns=cat_features,
        drop_cols=cat_drop,
    )
    cat_proba = cat_model.predict_proba(X_cat)[:, 1]
    print(f"    CatBoost tahmin tamam. Ortalama olasililik: {cat_proba.mean():.4f}")

    # -- XGBoost --
    from preprocessing import feature_engineering
    from preprocessing import fill_numeric_na  # noqa: F401

    df_fe = feature_engineering(test_df, training_stats=xgb_medians)

    # Eksik sayisal degerleri doldur
    for col in df_fe.select_dtypes(include=["float64", "int64"]).columns:
        if df_fe[col].isna().any():
            df_fe[col] = df_fe[col].fillna(xgb_medians.get(col, 0))

    df_ohe = pd.get_dummies(df_fe, drop_first=False)

    # Eksik sutunlari sifirla, fazlalari kaldir
    for col in xgb_columns:
        if col not in df_ohe.columns:
            df_ohe[col] = 0
    X_xgb = df_ohe[xgb_columns]

    xgb_proba = xgb_model.predict_proba(X_xgb)[:, 1]
    print(f"    XGBoost tahmin tamam. Ortalama olasililik: {xgb_proba.mean():.4f}")

    return cat_proba, xgb_proba


# --------------------------------------------------------------------- #
# Esik noktasini bul                                                     #
# --------------------------------------------------------------------- #
def _threshold_point_roc(fpr, tpr, thresholds, threshold_value):
    """ROC egrisinde esige en yakin (fpr, tpr) noktasini dondurur."""
    idx = np.argmin(np.abs(thresholds - threshold_value))
    return fpr[idx], tpr[idx]


def _threshold_point_pr(precision, recall, thresholds, threshold_value):
    """PR egrisinde esige en yakin (recall, precision) noktasini dondurur."""
    idx = np.argmin(np.abs(thresholds - threshold_value))
    return recall[idx], precision[idx]


# --------------------------------------------------------------------- #
# Grafik: ROC Egrisi                                                     #
# --------------------------------------------------------------------- #
def plot_roc(
    y_true,
    cat_proba, cat_threshold,
    xgb_proba, xgb_threshold,
    save_path,
):
    print("[4/5] ROC egrisi ciziliyor...")

    fpr_cat, tpr_cat, thr_cat = roc_curve(y_true, cat_proba)
    auc_cat = auc(fpr_cat, tpr_cat)

    fpr_xgb, tpr_xgb, thr_xgb = roc_curve(y_true, xgb_proba)
    auc_xgb = auc(fpr_xgb, tpr_xgb)

    # Esik noktalari
    fx_cat, ty_cat = _threshold_point_roc(fpr_cat, tpr_cat, thr_cat, cat_threshold)
    fx_xgb, ty_xgb = _threshold_point_roc(fpr_xgb, tpr_xgb, thr_xgb, xgb_threshold)

    fig, ax = plt.subplots(figsize=(9, 6))

    # Egri cizgileri
    ax.plot(
        fpr_cat, tpr_cat,
        color=CAT_COLOR, linewidth=2,
        label=f"CatBoost (AUC = {auc_cat:.4f})",
    )
    ax.plot(
        fpr_xgb, tpr_xgb,
        color=XGB_COLOR, linewidth=2,
        label=f"XGBoost (AUC = {auc_xgb:.4f})",
    )

    # Esik yildizlari
    ax.scatter(
        fx_cat, ty_cat,
        color=CAT_COLOR, marker="*", s=150, zorder=5,
        label=f"CatBoost esigi ({cat_threshold:.2f})",
    )
    ax.scatter(
        fx_xgb, ty_xgb,
        color=XGB_COLOR, marker="*", s=150, zorder=5,
        label=f"XGBoost esigi ({xgb_threshold:.2f})",
    )

    # Rastgele tahmin referans cizgisi
    ax.plot(
        [0, 1], [0, 1],
        color="gray", linestyle="--", linewidth=1.5,
        label="Rastgele Tahmin",
    )

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("Yanlis Pozitif Orani (1 - Ozgulluk)", fontsize=12)
    ax.set_ylabel("Dogru Pozitif Orani (Duyarlilik)", fontsize=12)
    ax.set_title("ROC Egrisi — Model Karsilastirmasi", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"    CatBoost AUC : {auc_cat:.4f}")
    print(f"    XGBoost  AUC : {auc_xgb:.4f}")
    return auc_cat, auc_xgb


# --------------------------------------------------------------------- #
# Grafik: PR Egrisi                                                      #
# --------------------------------------------------------------------- #
def plot_pr(
    y_true,
    cat_proba, cat_threshold,
    xgb_proba, xgb_threshold,
    save_path,
):
    print("[5/5] PR egrisi ciziliyor...")

    churn_rate = y_true.mean()

    prec_cat, rec_cat, thr_cat = precision_recall_curve(y_true, cat_proba)
    ap_cat = average_precision_score(y_true, cat_proba)

    prec_xgb, rec_xgb, thr_xgb = precision_recall_curve(y_true, xgb_proba)
    ap_xgb = average_precision_score(y_true, xgb_proba)

    # precision_recall_curve: thresholds uzunlugu prec/rec'den 1 eksik
    rx_cat, py_cat = _threshold_point_pr(
        prec_cat[:-1], rec_cat[:-1], thr_cat, cat_threshold
    )
    rx_xgb, py_xgb = _threshold_point_pr(
        prec_xgb[:-1], rec_xgb[:-1], thr_xgb, xgb_threshold
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        rec_cat, prec_cat,
        color=CAT_COLOR, linewidth=2,
        label=f"CatBoost (AP = {ap_cat:.4f})",
    )
    ax.plot(
        rec_xgb, prec_xgb,
        color=XGB_COLOR, linewidth=2,
        label=f"XGBoost (AP = {ap_xgb:.4f})",
    )

    # Esik yildizlari  (PR'da x=recall, y=precision)
    ax.scatter(
        rx_cat, py_cat,
        color=CAT_COLOR, marker="*", s=150, zorder=5,
        label=f"CatBoost esigi ({cat_threshold:.2f})",
    )
    ax.scatter(
        rx_xgb, py_xgb,
        color=XGB_COLOR, marker="*", s=150, zorder=5,
        label=f"XGBoost esigi ({xgb_threshold:.2f})",
    )

    # Taban deger (churn orani)
    ax.axhline(
        y=churn_rate,
        color="gray", linestyle="--", linewidth=1.5,
        label=f"Taban Deger (Churn Orani = {churn_rate:.3f})",
    )

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("Duyarlilik (Recall)", fontsize=12)
    ax.set_ylabel("Hassasiyet (Precision)", fontsize=12)
    ax.set_title(
        "Hassasiyet-Duyarlilik Egrisi — Model Karsilastirmasi",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"    CatBoost AP  : {ap_cat:.4f}")
    print(f"    XGBoost  AP  : {ap_xgb:.4f}")
    return ap_cat, ap_xgb


# --------------------------------------------------------------------- #
# Ana akis                                                               #
# --------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="ROC ve PR egri grafikleri olusturur."
    )
    parser.add_argument(
        "csv", nargs="?", default=None,
        help="IBM Telco formatinda CSV dosyasi yolu (Churn sutunu gerekli)",
    )
    parser.add_argument(
        "--path", default=None,
        help="--path ile CSV yolu belirtme alternatifi",
    )
    args = parser.parse_args()

    cli_path = args.path or args.csv

    # Artifact yukle
    (
        cat_model, cat_threshold, cat_medians, cat_features, cat_drop,
        xgb_model, xgb_threshold, xgb_columns, xgb_medians,
    ) = load_artifacts()

    # CSV bul
    csv_path = resolve_csv_path(cli_path)

    # Veri yukle ve bol
    test_df, y_true = load_and_split(csv_path)

    # Tahminler
    cat_proba, xgb_proba = get_predictions(
        test_df,
        cat_model, cat_medians, cat_features, cat_drop,
        xgb_model, xgb_columns, xgb_medians,
    )

    # Cikti dizini olustur
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    roc_path = os.path.join(SCRIPT_DIR, OUTPUT_DIR, "roc_curve.png")
    pr_path  = os.path.join(SCRIPT_DIR, OUTPUT_DIR, "pr_curve.png")

    # Grafikler
    plot_roc(
        y_true,
        cat_proba, cat_threshold,
        xgb_proba, xgb_threshold,
        roc_path,
    )
    plot_pr(
        y_true,
        cat_proba, cat_threshold,
        xgb_proba, xgb_threshold,
        pr_path,
    )

    print("\n" + "=" * 60)
    print("Grafikler kaydedildi:")
    print(f"  ROC egrisi : {os.path.abspath(roc_path)}")
    print(f"  PR  egrisi : {os.path.abspath(pr_path)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
