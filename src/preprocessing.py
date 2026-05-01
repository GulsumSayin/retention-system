"""
preprocessing.py
Veri ön işleme, özellik mühendisliği ve model girdisi hazırlama modülü.

Veri Sızıntısı Notu (tez için kritik):
  HighRiskProfile ve ShortTenure_HighCharge özellikleri eşik değerleri olarak
  sırasıyla MonthlyCharges medyanı ve 75. yüzdelik dilimini kullanır.
  Bu değerler çıkarım zamanında *o anki batch'ten* hesaplanırsa:
    - Aynı müşteri farklı batch'lerde farklı özellik değerleri alabilir.
    - Eğitim seti dağılımından sapma (train/inference distribution shift) oluşur.
  Düzeltme: feature_engineering() artık isteğe bağlı training_stats parametresi
  alır. Eğitim seti istatistikleri sağlandığında batch istatistiği kullanılmaz.
  training_stats sağlanmazsa batch istatistiği kullanılır ve bir uyarı loglanır.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Özellik Mühendisliği
# ---------------------------------------------------------------------------

def feature_engineering(
    data: pd.DataFrame,
    training_stats: dict | None = None,
) -> pd.DataFrame:
    """
    Ham müşteri verisinden model özellikleri üretir.

    Parametreler
    ------------
    data           : Ham DataFrame (IBM Telco formatı)
    training_stats : Eğitim seti istatistikleri (ör. train_medians.pkl içeriği).
                     HighRiskProfile ve ShortTenure_HighCharge eşikleri için
                     "MonthlyCharges" (medyan) ve "MonthlyCharges_q75" (Q75)
                     anahtarları beklenir. Sağlanmazsa batch istatistikleri
                     kullanılır ve logger.warning tetiklenir.

    Döndürür
    --------
    pd.DataFrame : Özellikler eklenmiş DataFrame (customerID ve Churn kaldırılmış)
    """
    df = data.copy()

    for col in ["customerID", "Churn"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # ------------------------------------------------------------------ #
    # Tenure tabanlı özellikler
    # ------------------------------------------------------------------ #
    tenure_safe = df["tenure"].replace(0, 1)

    if "TotalCharges" in df.columns:
        df["AvgMonthlySpend"] = df["TotalCharges"] / tenure_safe

    df["IsNewCustomer"]   = (df["tenure"] <= 12).astype(int)
    df["IsLoyalCustomer"] = (df["tenure"] >= 24).astype(int)
    df["IsMonthToMonth"]  = (
        (df["Contract"] == "Month-to-month").astype(int)
        if "Contract" in df.columns else 0
    )

    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0_12", "13_24", "25_48", "49_72"],
    )

    # ------------------------------------------------------------------ #
    # Sözleşme skoru
    # ------------------------------------------------------------------ #
    if "Contract" in df.columns:
        df["ContractScore"] = df["Contract"].map(
            {"Month-to-month": 2, "One year": 1, "Two year": 0}
        )

    # ------------------------------------------------------------------ #
    # Ödeme yöntemi özellikleri
    # ------------------------------------------------------------------ #
    if "PaymentMethod" in df.columns:
        df["UsesAutoPayment"] = df["PaymentMethod"].isin(
            ["Bank transfer (automatic)", "Credit card (automatic)"]
        ).astype(int)

        df["PaymentRisk"] = df["PaymentMethod"].map(
            {
                "Electronic check":             2,
                "Mailed check":                 1,
                "Bank transfer (automatic)":    0,
                "Credit card (automatic)":      0,
            }
        )

    # ------------------------------------------------------------------ #
    # Servis yoğunluğu
    # ------------------------------------------------------------------ #
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    existing_service_cols = [c for c in service_cols if c in df.columns]

    def count_services(row: pd.Series) -> int:
        return sum(
            1 for c in existing_service_cols
            if str(row[c]) not in {"No", "No internet service", "No phone service"}
        )

    df["NumServices"] = (
        df[existing_service_cols].apply(count_services, axis=1)
        if existing_service_cols else 0
    )

    df["Tenure_Charges"]    = df["tenure"] * df["MonthlyCharges"]
    df["ServiceIntensity"]  = df["NumServices"] / (df["tenure"] + 1)
    df["ChargesPerService"] = df["MonthlyCharges"] / (df["NumServices"] + 1)

    # ------------------------------------------------------------------ #
    # Risk profil bayrakları — eğitim istatistikleri kullan (sızıntı önleme)
    # ------------------------------------------------------------------ #
    monthly_median, monthly_q75 = _resolve_monthly_thresholds(df, training_stats)

    df["HighRiskProfile"] = (
        (df["tenure"] < 12) &
        (df["MonthlyCharges"] > monthly_median)
    ).astype(int)

    df["ShortTenure_HighCharge"] = (
        (df["tenure"] <= 12) &
        (df["MonthlyCharges"] >= monthly_q75)
    ).astype(int)

    # ------------------------------------------------------------------ #
    # Koruma servisi eksikliği
    # ------------------------------------------------------------------ #
    protection_cols = ["OnlineSecurity", "TechSupport", "DeviceProtection"]
    existing_protection_cols = [c for c in protection_cols if c in df.columns]

    def no_protection_flag(row: pd.Series) -> int:
        return sum(1 for c in existing_protection_cols if str(row[c]) == "No")

    df["NoProtectionFlag"] = (
        df[existing_protection_cols].apply(no_protection_flag, axis=1)
        if existing_protection_cols else 0
    )

    # ------------------------------------------------------------------ #
    # Kombinasyon özellikleri
    # ------------------------------------------------------------------ #
    if {"Contract", "PaymentMethod"}.issubset(df.columns):
        df["Contract_Payment"] = (
            df["Contract"].astype(str) + "_" + df["PaymentMethod"].astype(str)
        )
        df["MonthToMonth_ElectronicCheck"] = (
            (df["Contract"] == "Month-to-month") &
            (df["PaymentMethod"] == "Electronic check")
        ).astype(int)
    else:
        df["MonthToMonth_ElectronicCheck"] = 0

    if {"Contract", "InternetService"}.issubset(df.columns):
        df["Contract_Internet"] = (
            df["Contract"].astype(str) + "_" + df["InternetService"].astype(str)
        )

    if {"Partner", "Dependents"}.issubset(df.columns):
        df["FamilyRisk"] = (
            (df["Partner"] == "No").astype(int) +
            (df["Dependents"] == "No").astype(int)
        )
    else:
        df["FamilyRisk"] = 0

    # ------------------------------------------------------------------ #
    # Sonsuz / NaN temizliği
    # ------------------------------------------------------------------ #
    for col in ["AvgMonthlySpend", "ChargesPerService", "ServiceIntensity"]:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df


def _resolve_monthly_thresholds(
    df: pd.DataFrame,
    training_stats: dict | None,
) -> tuple[float, float]:
    """
    HighRiskProfile ve ShortTenure_HighCharge için MonthlyCharges eşiklerini döner.

    Öncelik:
      1. training_stats["MonthlyCharges"]      → medyan (eğitim setinden)
      2. training_stats["MonthlyCharges_q75"]  → Q75 (eğitim setinden)
      3. Yukarıdakiler yoksa batch istatistikleri + uyarı logu

    Tez notu:
      Eğitim sırasında train_medians.pkl dosyasına hem "MonthlyCharges" (medyan)
      hem de "MonthlyCharges_q75" anahtarları eklenmelidir.
    """
    if training_stats:
        median = training_stats.get("MonthlyCharges")
        q75    = training_stats.get("MonthlyCharges_q75")

        if median is None:
            logger.warning(
                "training_stats içinde 'MonthlyCharges' medyanı bulunamadı; "
                "batch medyanı kullanılıyor (veri sızıntısı riski)."
            )
            median = float(df["MonthlyCharges"].median())
        if q75 is None:
            logger.warning(
                "training_stats içinde 'MonthlyCharges_q75' bulunamadı; "
                "batch Q75 kullanılıyor (veri sızıntısı riski)."
            )
            q75 = float(df["MonthlyCharges"].quantile(0.75))

        return float(median), float(q75)

    logger.warning(
        "training_stats sağlanmadı. HighRiskProfile ve ShortTenure_HighCharge "
        "için batch istatistikleri kullanılıyor. "
        "Bu durum train/inference dağılım uyuşmazlığına yol açabilir."
    )
    return (
        float(df["MonthlyCharges"].median()),
        float(df["MonthlyCharges"].quantile(0.75)),
    )


# ---------------------------------------------------------------------------
# Eksik değer doldurma
# ---------------------------------------------------------------------------

def fill_numeric_na(df: pd.DataFrame, train_medians: dict) -> pd.DataFrame:
    """
    Sayısal sütunlardaki eksik değerleri eğitim medyanlarıyla doldurur.

    train_medians'ta bulunmayan türetilmiş sütunlar (AvgMonthlySpend,
    ChargesPerService, ServiceIntensity vb.) için 0 ile doldurma uygulanır;
    bu değerler zaten oran/yoğunluk sütunları olduğundan 0 semantik olarak
    "veri yok / hesaplanamadı" anlamına gelir.
    """
    out = df.copy()
    for col in out.select_dtypes(include=["int64", "float64"]).columns:
        if not out[col].isna().any():
            continue
        if col in train_medians:
            out[col] = out[col].fillna(train_medians[col])
        else:
            out[col] = out[col].fillna(0)
            logger.debug(
                "fill_numeric_na: '%s' train_medians'ta yok; 0 ile dolduruldu.", col
            )
    return out


# ---------------------------------------------------------------------------
# XGBoost girdisi
# ---------------------------------------------------------------------------

def prepare_xgb_input(
    raw_df:        pd.DataFrame,
    train_medians: dict,
    model_columns: list,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    XGBoost için one-hot encode edilmiş, model sütunlarıyla hizalanmış DataFrame döner.

    Döndürür
    --------
    (X_ready, df_fe) : Model girdisi ve özellik mühendisliği çıktısı
    """
    df_fe  = feature_engineering(raw_df, training_stats=train_medians)
    df_fe  = fill_numeric_na(df_fe, train_medians)
    df_ohe = pd.get_dummies(df_fe, drop_first=False)

    for col in model_columns:
        if col not in df_ohe.columns:
            df_ohe[col] = 0

    return df_ohe[model_columns], df_fe


# ---------------------------------------------------------------------------
# CatBoost girdisi
# ---------------------------------------------------------------------------

def prepare_cat_input(
    raw_df:          pd.DataFrame,
    train_medians:   dict,
    feature_columns: list,
    drop_cols:       list,
) -> pd.DataFrame:
    """
    CatBoost için native kategorik desteğiyle hazırlanmış, feature_columns
    sırasına hizalanmış DataFrame döner.
    """
    df_fe = feature_engineering(raw_df, training_stats=train_medians)
    df_fe = fill_numeric_na(df_fe, train_medians)
    df_fe = df_fe.drop(columns=drop_cols, errors="ignore")

    for col in feature_columns:
        if col not in df_fe.columns:
            df_fe[col] = 0

    return df_fe[feature_columns]
