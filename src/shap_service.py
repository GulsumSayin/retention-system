"""
shap_service.py
SHAP (SHapley Additive exPlanations) tabanlı model açıklanabilirlik modülü.

Tez Konumu:
  Bu modül "açıklanabilir yapay zekâ (XAI)" katmanını oluşturur.
  agents.py generate_reason() fonksiyonunun post-hoc açıklanabilir alternatifi;
  aksiyon gerekçeleri modelin öğrendiği feature importance'a dayanır.

Akademik Katkı:
  - Gerekçeler veri odaklı ve model tutarlıdır (post-hoc explanation)
  - Her müşteri için bireysel SHAP değerleri hesaplanır (local explanation)
    Kaynak: Lundberg & Lee, "A Unified Approach to Interpreting Model
    Predictions", NeurIPS 2017.
  - Global feature önem sırası portföy düzeyinde analiz sağlar
  - agents.py generate_reason() ile doğrudan entegre olur

Yöntem:
  CatBoost için shap.TreeExplainer kullanılır (native destek, hızlı).
  XGBoost için de aynı sınıf çalışır.

DRY Notu:
  Kural tabanlı fallback gerekçesi (rule_based_reason) agents.py'de tek
  noktada tanımlıdır; bu modül onu import eder. Önceki sürümdeki _fallback_reason
  kopyası kaldırılmıştır.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from agents import rule_based_reason

logger = logging.getLogger(__name__)

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    logger.warning("shap paketi bulunamadı. `pip install shap` çalıştırın.")


# ---------------------------------------------------------------------------
# Feature açıklama sözlüğü — teknik isimler → Türkçe iş dili
# ---------------------------------------------------------------------------
_FEATURE_LABELS: dict[str, str] = {
    "MonthlyCharges":                   "yüksek aylık ücret",
    "tenure":                           "kısa müşteri süresi",
    "IsMonthToMonth":                   "aylık sözleşme",
    "Contract_Month-to-month":          "aylık sözleşme",
    "UsesAutoPayment":                  "otomatik ödeme kullanmıyor",
    "NoProtectionFlag":                 "koruma servisleri zayıf",
    "ServiceIntensity":                 "düşük servis yoğunluğu",
    "NumServices":                      "az servis kullanımı",
    "TechSupport_No":                   "teknik destek yok",
    "OnlineSecurity_No":                "online güvenlik yok",
    "DeviceProtection_No":              "cihaz koruması yok",
    "InternetService_Fiber optic":      "fiber internet + yüksek ücret",
    "PaymentMethod_Electronic check":   "elektronik çek ödeme riski",
    "HighRiskProfile":                  "yüksek risk profili",
    "ShortTenure_HighCharge":           "kısa süre + yüksek ücret",
    "MonthToMonth_ElectronicCheck":     "aylık sözleşme + elektronik çek",
    "FamilyRisk":                       "aile bağı zayıf",
    "ContractScore":                    "sözleşme riski",
    "ChargesPerService":                "servis başına yüksek ücret",
    "TenureGroup_0_12":                 "yeni müşteri (0-12 ay)",
    "IsNewCustomer":                    "yeni müşteri",
    "AvgMonthlySpend":                  "ortalama aylık harcama",
}


def _label(feature: str) -> str:
    """Feature adını Türkçe iş diline çevirir; eşleşme yoksa orijinal adı döner."""
    for key, val in _FEATURE_LABELS.items():
        if key.lower() in feature.lower():
            return val
    return feature


# ===========================================================================
# ShapService
# ===========================================================================

class ShapService:
    """
    SHAP TreeExplainer tabanlı bireysel ve portföy düzeyinde açıklanabilirlik.

    Kullanım:
        svc = ShapService()
        svc.fit(model, X_background)        # bir kez başlat
        reason = svc.explain_row(row, cols)  # bireysel gerekçe
        fig    = svc.plot_shap_bar(X_df)     # global önem grafiği
    """

    def __init__(self) -> None:
        self.explainer:    Optional[object] = None
        self.feature_names: list            = []
        self._fitted:       bool            = False

    # -----------------------------------------------------------------------
    # Fit
    # -----------------------------------------------------------------------

    def fit(self, model, X_background: pd.DataFrame) -> "ShapService":
        """
        SHAP TreeExplainer'ı verilen model ile başlatır.

        Parametreler
        ------------
        model        : CatBoost veya XGBoost model nesnesi
        X_background : Referans verisi (aday havuzu veya örneklenmiş alt küme)
        """
        if not _SHAP_AVAILABLE:
            logger.warning("SHAP yüklü değil, fit atlandı.")
            return self

        try:
            self.explainer     = shap.TreeExplainer(model)
            self.feature_names = list(X_background.columns)
            self._fitted       = True
            logger.info("ShapService hazır — %d feature.", len(self.feature_names))
        except Exception as exc:
            logger.error("ShapService.fit hatası: %s", exc)
        return self

    # -----------------------------------------------------------------------
    # Bireysel açıklama
    # -----------------------------------------------------------------------

    def explain_row(
        self,
        row:          pd.Series,
        feature_cols: list,
        top_n:        int = 3,
    ) -> str:
        """
        Tek müşteri için SHAP değerlerine dayalı Türkçe gerekçe üretir.
        agents.py generate_reason() ile aynı arayüze sahiptir.

        Fallback:
          SHAP hesaplanamadığında agents.rule_based_reason() çağrılır
          (DRY — kural tek noktada tanımlı).
        """
        if not self._fitted or self.explainer is None:
            return rule_based_reason(row)

        try:
            available = [c for c in feature_cols if c in row.index]
            X_row     = pd.DataFrame([row[available]], columns=available)
            shap_vals = self.explainer.shap_values(X_row)

            # CatBoost binary: liste döner; XGBoost: 2D array
            if isinstance(shap_vals, list):
                vals = shap_vals[1][0]
            else:
                vals = shap_vals[0] if shap_vals.ndim == 1 else shap_vals[0]

            # Churn'e en çok pozitif katkı yapan feature'lar
            indices   = np.argsort(np.abs(vals))[::-1][:top_n]
            top_feats = [available[i] for i in indices if vals[i] > 0]

            if not top_feats:
                return "Dengeli müşteri profili — belirgin SHAP katkısı yok."

            return ", ".join(_label(f) for f in top_feats)

        except Exception as exc:
            logger.error("explain_row hatası: %s", exc)
            return rule_based_reason(row)

    # -----------------------------------------------------------------------
    # Portföy SHAP değerleri
    # -----------------------------------------------------------------------

    def compute_shap_values(self, X_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Tüm aday havuzu için SHAP değerlerini hesaplar.
        Döndürülen DataFrame: satırlar müşteri, sütunlar feature SHAP değeri.
        """
        if not self._fitted or self.explainer is None:
            return None
        try:
            available = [c for c in self.feature_names if c in X_df.columns]
            shap_vals = self.explainer.shap_values(X_df[available])

            vals = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
            return pd.DataFrame(vals, columns=available, index=X_df.index)
        except Exception as exc:
            logger.error("compute_shap_values hatası: %s", exc)
            return None

    # -----------------------------------------------------------------------
    # Global feature önem sırası
    # -----------------------------------------------------------------------

    def global_importance(self, X_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Portföy düzeyinde ortalama |SHAP| değerlerine göre feature önem tablosu.
        Tez tablosu ve görselleştirme için kullanılabilir.

        Döndürür
        --------
        pd.DataFrame : feature, mean_abs_shap, label sütunları
        """
        shap_df = self.compute_shap_values(X_df)
        if shap_df is None:
            return pd.DataFrame()

        importance = (
            shap_df.abs()
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        importance.columns = ["feature", "mean_abs_shap"]
        importance["label"]         = importance["feature"].apply(_label)
        importance["mean_abs_shap"] = importance["mean_abs_shap"].round(4)
        return importance

    # -----------------------------------------------------------------------
    # Plotly SHAP özet grafiği
    # -----------------------------------------------------------------------

    def plot_shap_bar(
        self,
        X_df:  pd.DataFrame,
        top_n: int = 10,
        title: str = "SHAP Global Feature Önemi",
    ):
        """Plotly yatay bar chart olarak SHAP global feature önemini döner."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.error("plotly bulunamadı. `pip install plotly` çalıştırın.")
            return None

        importance = self.global_importance(X_df, top_n=top_n)
        if importance.empty:
            return None

        importance = importance.sort_values("mean_abs_shap", ascending=True)

        fig = go.Figure(go.Bar(
            x=importance["mean_abs_shap"],
            y=importance["label"],
            orientation="h",
            marker=dict(
                color=importance["mean_abs_shap"],
                colorscale=[[0, "#bfdbfe"], [1, "#1d4ed8"]],
                showscale=False,
            ),
            text=importance["mean_abs_shap"].apply(lambda v: f"{v:.4f}"),
            textposition="outside",
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        fig.update_layout(
            height=350,
            paper_bgcolor="white",
            plot_bgcolor="white",
            title=dict(text=title, font_size=14, font_color="#0f172a", x=0.0),
            xaxis=dict(
                title="Ortalama |SHAP| Değeri",
                showgrid=True,
                gridcolor="#f1f5f9",
                zeroline=False,
            ),
            yaxis=dict(showgrid=False, tickfont_size=11),
            margin=dict(t=60, b=40, l=10, r=60),
        )
        return fig

    # -----------------------------------------------------------------------
    # Bireysel SHAP waterfall verisi
    # -----------------------------------------------------------------------

    def waterfall_data(
        self,
        row:          pd.Series,
        feature_cols: list,
        top_n:        int = 8,
    ) -> Optional[pd.DataFrame]:
        """
        Tek müşteri için waterfall grafiğine hazır SHAP veri tablosu.
        Streamlit expander içinde gösterilebilir.
        """
        if not self._fitted or self.explainer is None:
            return None
        try:
            available = [c for c in feature_cols if c in row.index]
            X_row     = pd.DataFrame([row[available]], columns=available)
            shap_vals = self.explainer.shap_values(X_row)

            vals = shap_vals[1][0] if isinstance(shap_vals, list) else (
                shap_vals[0] if shap_vals.ndim == 1 else shap_vals[0]
            )

            df = pd.DataFrame({
                "feature":    available,
                "shap_value": vals,
                "label":      [_label(f) for f in available],
            })
            df = df.reindex(df["shap_value"].abs().sort_values(ascending=False).index)
            df = df.head(top_n).reset_index(drop=True)
            df["direction"] = df["shap_value"].apply(
                lambda v: "Churn riskini artırıyor" if v > 0 else "Churn riskini azaltıyor"
            )
            return df
        except Exception as exc:
            logger.error("waterfall_data hatası: %s", exc)
            return None
