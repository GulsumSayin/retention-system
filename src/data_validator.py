"""
data_validator.py
Giriş Veri Doğrulama Katmanı

Motivasyon:
  Kullanıcı yanlış format CSV yüklediğinde sistem belirsiz bir hatayla
  çökmeye bırakılmamalı; anlamlı, aksiyona yönelik hata mesajları verilmelidir.
  Production sistemlerde "Fail Fast" prensibi: veri kalite sorunları boru
  hattına girmeden önce yakalanır.

Doğrulama Katmanları:
  1. Şema Kontrolü: zorunlu sütunların varlığı
  2. Tip Kontrolü: sayısal sütunların dönüşebilirliği
  3. Değer Aralığı: makul aralık dışı değerler
  4. Eksik Değer Oranı: belirli eşiği aşan sütunlar
  5. Aykırı Değer Tespiti: IQR yöntemiyle
  6. Çift Satır Kontrolü: customerID tekrarı

Kullanım:
    validator = DataValidator()
    report    = validator.validate(df)
    if not report.is_valid:
        st.error(report.format_errors())
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IBM Telco formatı şeması
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS: list[str] = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "Contract", "PaymentMethod", "InternetService",
]

RECOMMENDED_COLUMNS: list[str] = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "PaperlessBilling",
]

NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "tenure":         (0, 120),
    "MonthlyCharges": (0, 500),
    "TotalCharges":   (0, 10_000),
    "SeniorCitizen":  (0, 1),
}

CATEGORICAL_DOMAINS: dict[str, set[str]] = {
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaymentMethod": {
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    },
    "InternetService": {"DSL", "Fiber optic", "No"},
}

MAX_MISSING_RATIO: float = 0.30   # %30 üzeri eksik → uyarı


# ===========================================================================
# Validation Raporu
# ===========================================================================

@dataclass
class ValidationReport:
    is_valid:   bool = True
    errors:     list[str] = field(default_factory=list)
    warnings:   list[str] = field(default_factory=list)
    info:       list[str] = field(default_factory=list)
    row_count:  int = 0
    col_count:  int = 0

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False
        logger.error("Validasyon hatası: %s", msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.warning("Validasyon uyarısı: %s", msg)

    def add_info(self, msg: str) -> None:
        self.info.append(msg)

    def format_errors(self) -> str:
        """Streamlit'te gösterilecek hata özeti."""
        lines = [f"**Veri Doğrulama Hatası** — {len(self.errors)} hata bulundu:"]
        for e in self.errors:
            lines.append(f"- {e}")
        return "\n".join(lines)

    def format_warnings(self) -> str:
        if not self.warnings:
            return ""
        lines = [f"**Veri Uyarıları** — {len(self.warnings)} uyarı:"]
        for w in self.warnings:
            lines.append(f"- {w}")
        return "\n".join(lines)


# ===========================================================================
# DataValidator
# ===========================================================================

class DataValidator:
    """
    IBM Telco formatındaki müşteri verisi için kapsamlı doğrulama servisi.

    Kullanım:
        report = DataValidator().validate(df)
        if not report.is_valid:
            raise ValueError(report.format_errors())
    """

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """
        Veriyi tüm kontrol katmanlarından geçirir ve ValidationReport döner.

        Parametreler
        ------------
        df : Kullanıcıdan yüklenen ham DataFrame

        Döndürür
        --------
        ValidationReport : is_valid=False ise pipeline durdurulmalı
        """
        report           = ValidationReport()
        report.row_count = len(df)
        report.col_count = len(df.columns)

        report.add_info(f"Yüklenen veri: {report.row_count:,} satır, {report.col_count} sütun.")

        if report.row_count == 0:
            report.add_error("DataFrame boş — satır yok.")
            return report

        self._check_required_columns(df, report)
        if not report.is_valid:
            return report  # Zorunlu sütun yoksa devam etme

        self._check_recommended_columns(df, report)
        self._check_numeric_types(df, report)
        self._check_value_bounds(df, report)
        self._check_categorical_domains(df, report)
        self._check_missing_ratios(df, report)
        self._check_outliers(df, report)
        self._check_duplicate_customers(df, report)
        self._check_data_stats(df, report)

        if report.is_valid:
            logger.info("Veri doğrulama başarılı: %d satır.", report.row_count)
        else:
            logger.error("Veri doğrulama başarısız: %d hata.", len(report.errors))

        return report

    # -----------------------------------------------------------------------
    # Kontrol Katmanları
    # -----------------------------------------------------------------------

    @staticmethod
    def _check_required_columns(df: pd.DataFrame, report: ValidationReport) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            report.add_error(
                f"Zorunlu sütunlar eksik: {missing}. "
                f"IBM Telco formatı beklenmektedir."
            )
        else:
            report.add_info("Zorunlu sütunlar mevcut.")

    @staticmethod
    def _check_recommended_columns(df: pd.DataFrame, report: ValidationReport) -> None:
        missing = [c for c in RECOMMENDED_COLUMNS if c not in df.columns]
        if missing:
            report.add_warning(
                f"{len(missing)} önerilen sütun eksik: {missing[:5]}{'...' if len(missing) > 5 else ''}. "
                f"Feature engineering bazı özellikleri üretemeyebilir."
            )

    @staticmethod
    def _check_numeric_types(df: pd.DataFrame, report: ValidationReport) -> None:
        for col in ["tenure", "MonthlyCharges"]:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    report.add_error(f"'{col}' sayısal olmalı, bulundu: {df[col].dtype}.")

        if "TotalCharges" in df.columns:
            coerced = pd.to_numeric(df["TotalCharges"], errors="coerce")
            n_fail  = coerced.isna().sum() - df["TotalCharges"].isna().sum()
            if n_fail > 0:
                report.add_warning(
                    f"TotalCharges: {n_fail:,} satır sayıya dönüştürülemiyor "
                    f"(boşluk/string). Otomatik coerce uygulanacak."
                )

    @staticmethod
    def _check_value_bounds(df: pd.DataFrame, report: ValidationReport) -> None:
        for col, (lo, hi) in NUMERIC_BOUNDS.items():
            if col not in df.columns:
                continue
            num = pd.to_numeric(df[col], errors="coerce")
            n_oob = ((num < lo) | (num > hi)).sum()
            if n_oob > 0:
                report.add_warning(
                    f"'{col}': {n_oob:,} satır beklenen aralık dışında "
                    f"([{lo}, {hi}]). Aykırı değer kontrolü yapın."
                )

    @staticmethod
    def _check_categorical_domains(df: pd.DataFrame, report: ValidationReport) -> None:
        for col, domain in CATEGORICAL_DOMAINS.items():
            if col not in df.columns:
                continue
            found   = set(df[col].dropna().unique())
            unknown = found - domain
            if unknown:
                report.add_warning(
                    f"'{col}': Bilinmeyen kategoriler: {unknown}. "
                    f"Beklenen: {domain}."
                )

    @staticmethod
    def _check_missing_ratios(df: pd.DataFrame, report: ValidationReport) -> None:
        for col in df.columns:
            ratio = df[col].isna().mean()
            if ratio > MAX_MISSING_RATIO:
                report.add_warning(
                    f"'{col}': %{ratio*100:.1f} eksik değer "
                    f"(eşik: %{MAX_MISSING_RATIO*100:.0f})."
                )

        total_missing = df.isna().sum().sum()
        if total_missing > 0:
            report.add_info(f"Toplam {total_missing:,} eksik değer tespit edildi.")

    @staticmethod
    def _check_outliers(df: pd.DataFrame, report: ValidationReport) -> None:
        for col in ["MonthlyCharges", "TotalCharges", "tenure"]:
            if col not in df.columns:
                continue
            num = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(num) < 4:
                continue
            Q1, Q3   = num.quantile(0.25), num.quantile(0.75)
            IQR      = Q3 - Q1
            n_out    = ((num < Q1 - 3 * IQR) | (num > Q3 + 3 * IQR)).sum()
            if n_out > 0:
                report.add_warning(
                    f"'{col}': {n_out:,} aşırı aykırı değer (3×IQR kuralı). "
                    f"Veri kalitesini kontrol edin."
                )

    @staticmethod
    def _check_duplicate_customers(df: pd.DataFrame, report: ValidationReport) -> None:
        if "customerID" in df.columns:
            n_dup = df["customerID"].duplicated().sum()
            if n_dup > 0:
                report.add_warning(
                    f"{n_dup:,} yinelenen customerID tespit edildi. "
                    f"Tahmin tutarsızlıklarına yol açabilir."
                )
        else:
            n_dup = df.duplicated().sum()
            if n_dup > 0:
                report.add_warning(f"{n_dup:,} tamamen yinelenen satır tespit edildi.")

    @staticmethod
    def _check_data_stats(df: pd.DataFrame, report: ValidationReport) -> None:
        if "tenure" in df.columns:
            avg_tenure = pd.to_numeric(df["tenure"], errors="coerce").mean()
            report.add_info(f"Ortalama müşteri süresi: {avg_tenure:.1f} ay.")
        if "MonthlyCharges" in df.columns:
            avg_mc = pd.to_numeric(df["MonthlyCharges"], errors="coerce").mean()
            report.add_info(f"Ortalama aylık ücret: {avg_mc:.2f} TL.")
        if "Contract" in df.columns:
            mtm_ratio = (df["Contract"] == "Month-to-month").mean()
            report.add_info(f"Aylık sözleşme oranı: %{mtm_ratio*100:.1f}.")
