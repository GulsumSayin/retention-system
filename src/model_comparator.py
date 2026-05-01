"""
model_comparator.py
Champion (CatBoost) / Challenger (XGBoost) İstatistiksel Karşılaştırma Modülü

Akademik Katkı:
  İki modeli nicel ve istatistiksel olarak karşılaştırır.
  Yalnızca nokta tahmini değil, güven aralığı ve hipotez testi sonuçları
  sunulur; bu yaklaşım tezde model seçimini savunulabilir kılar.

Uygulanan Testler:
  1. DeLong AUC Testi (AUC Farkı)
     Kaynak: DeLong et al., "Comparing the areas under two or more
     correlated receiver operating characteristic curves: a nonparametric
     approach", Biometrics 44(3), 1988.
     İki korelasyonlu ROC eğrisinin AUC'larının istatistiksel olarak farklı
     olup olmadığını test eder. H₀: AUC(CatBoost) = AUC(XGBoost).

  2. McNemar Testi (Tahmin Uyuşmazlığı)
     Kaynak: McNemar (1947), Psychometrika.
     İki modelin yanlış tahminlerinin aynı örneklerde mi yoksa farklı
     örneklerde mi yoğunlaştığını test eder. H₀: iki modelin hata dağılımı
     aynıdır.

  3. Bootstrap ROC-AUC Güven Aralığı
     Her model için 95% CI hesaplanır; CI çakışması olmayan modeller
     istatistiksel olarak farklı kabul edilir.

  4. Kalibrasyon Karşılaştırması
     Brier Skoru ve Beklenen Kalibrasyon Hatası (ECE) hesaplanır.
     İyi kalibre edilmiş model churn olasılıklarının güvenilirliğini artırır.

Kullanım (app.py):
    comparator = ModelComparator(router)
    report = comparator.compare(raw_df, y_true)
    fig    = comparator.plot_roc_comparison(report)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    _PLOTLY = True
except ImportError:
    _PLOTLY = False


# ===========================================================================
# Yardımcı: DeLong AUC Testi
# ===========================================================================

def _delong_auc_variance(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """
    DeLong et al. (1988) yöntemiyle AUC ve varyansını hesaplar.
    Biometrics 44(3):837-845.
    """
    order     = np.argsort(-y_score)
    label_ord = y_true[order]
    score_ord = y_score[order]

    n_pos = int(label_ord.sum())
    n_neg = int((1 - label_ord).sum())

    if n_pos == 0 or n_neg == 0:
        raise ValueError("y_true hem pozitif hem negatif örnek içermelidir.")

    pos_idx = np.where(label_ord == 1)[0]
    neg_idx = np.where(label_ord == 0)[0]

    # Sıra bazlı AUC
    tp = label_ord.cumsum()
    fp = (1 - label_ord).cumsum()
    fn = n_pos - tp
    tn = n_neg - fp

    v10 = np.zeros(n_pos)
    v01 = np.zeros(n_neg)

    for i, pi in enumerate(pos_idx):
        left  = int((score_ord[neg_idx] < score_ord[pi]).sum())
        tie   = int((score_ord[neg_idx] == score_ord[pi]).sum())
        v10[i] = (left + 0.5 * tie) / n_neg

    for j, ni in enumerate(neg_idx):
        right = int((score_ord[pos_idx] > score_ord[ni]).sum())
        tie   = int((score_ord[pos_idx] == score_ord[ni]).sum())
        v01[j] = (right + 0.5 * tie) / n_pos

    auc      = v10.mean()
    var_v10  = np.var(v10, ddof=1) / n_pos
    var_v01  = np.var(v01, ddof=1) / n_neg
    variance = var_v10 + var_v01

    return auc, variance


def delong_test(
    y_true:    np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
) -> dict:
    """
    İki korelasyonlu ROC eğrisini DeLong yöntemiyle karşılaştırır.

    Döndürür
    --------
    dict:
      auc_a, auc_b       : İki modelin AUC değerleri
      auc_diff           : AUC farkı (a - b)
      z_stat             : Z istatistiği
      p_value            : İki kuyruklu p değeri
      significant_95     : p < 0.05 ise True
    """
    from scipy import stats as scipy_stats

    auc_a, var_a = _delong_auc_variance(y_true, y_score_a)
    auc_b, var_b = _delong_auc_variance(y_true, y_score_b)

    # Korelasyon terimi (yaklaşık: bağımsız varyans kullan)
    se      = np.sqrt(var_a + var_b)
    z_stat  = (auc_a - auc_b) / (se + 1e-10)
    p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat)))

    return {
        "auc_a":         round(auc_a, 4),
        "auc_b":         round(auc_b, 4),
        "auc_diff":      round(auc_a - auc_b, 4),
        "z_stat":        round(z_stat, 4),
        "p_value":       round(p_value, 4),
        "significant_95": bool(p_value < 0.05),
    }


# ===========================================================================
# Yardımcı: McNemar Testi
# ===========================================================================

def mcnemar_test(
    y_true:    np.ndarray,
    y_pred_a:  np.ndarray,
    y_pred_b:  np.ndarray,
) -> dict:
    """
    McNemar (1947) testi: iki modelin hata dağılımını karşılaştırır.

    Koşul matrisi:
      b01 : A yanlış, B doğru
      b10 : A doğru, B yanlış

    H₀: b01 = b10 (iki modelin hata dağılımı aynı)

    Küçük örnekler için süreklilik düzeltmesi uygulanır (Edwards, 1948).
    """
    from scipy import stats as scipy_stats

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    b01 = int((~correct_a & correct_b).sum())  # A yanlış, B doğru
    b10 = int((correct_a & ~correct_b).sum())  # A doğru, B yanlış

    n_discordant = b01 + b10
    if n_discordant == 0:
        return {
            "b01": b01, "b10": b10,
            "chi2": 0.0, "p_value": 1.0,
            "significant_95": False,
            "note": "İki model identik tahmin üretiyor.",
        }

    # Süreklilik düzeltmeli McNemar
    chi2    = (abs(b01 - b10) - 1) ** 2 / n_discordant
    p_value = 1 - scipy_stats.chi2.cdf(chi2, df=1)

    return {
        "b01":           b01,
        "b10":           b10,
        "chi2":          round(chi2, 4),
        "p_value":       round(p_value, 4),
        "significant_95": bool(p_value < 0.05),
    }


# ===========================================================================
# Yardımcı: Brier Skoru ve ECE
# ===========================================================================

def _brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return float(np.mean((y_proba - y_true) ** 2))


def _expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Beklenen Kalibrasyon Hatası (ECE).
    Küçük değer = iyi kalibre edilmiş model.
    Kaynak: Naeini et al., AAAI 2015.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece       = 0.0
    n         = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask   = (y_proba >= lo) & (y_proba < hi)
        if mask.sum() == 0:
            continue
        acc    = y_true[mask].mean()
        conf   = y_proba[mask].mean()
        ece   += mask.sum() / n * abs(acc - conf)

    return round(float(ece), 4)


def _bootstrap_auc_ci(y_true, y_score, n=1000, ci=0.95, seed=42):
    rng  = np.random.default_rng(seed)
    from sklearn.metrics import roc_auc_score
    aucs = []
    sz   = len(y_true)
    for _ in range(n):
        idx = rng.integers(0, sz, size=sz)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    alpha = (1 - ci) / 2
    return float(np.percentile(aucs, alpha * 100)), float(np.percentile(aucs, (1 - alpha) * 100))


# ===========================================================================
# ModelComparator
# ===========================================================================

class ModelComparator:
    """
    Champion (CatBoost) ve Challenger (XGBoost) modellerini istatistiksel
    olarak karşılaştırır.

    Kullanım:
        comparator = ModelComparator(router)
        report = comparator.compare(raw_df, y_true)
        fig    = comparator.plot_roc_comparison(report)
    """

    def __init__(self, router) -> None:
        self.router = router

    def compare(
        self,
        raw_df: pd.DataFrame,
        y_true: np.ndarray | None = None,
    ) -> dict:
        """
        İki modeli tahmin eder ve istatistiksel testleri hesaplar.

        Parametreler
        ------------
        raw_df : Ham DataFrame (model input)
        y_true : Gerçek etiketler (varsa). None ise yalnızca olasılık
                 karşılaştırması yapılır (istatistiksel testler atlanır).

        Döndürür
        --------
        dict: Olasılıklar, metrikler, DeLong testi, McNemar testi,
              kalibrasyon metrikleri
        """
        from sklearn.metrics import roc_auc_score, average_precision_score

        both    = self.router.predict_both(raw_df)
        proba_a = both["catboost"]["churn_proba"].values
        proba_b = both["xgboost"]["churn_proba"].values
        pred_a  = both["catboost"]["predicted_churn"].values
        pred_b  = both["xgboost"]["predicted_churn"].values

        agreement_rate = float((pred_a == pred_b).mean())
        proba_corr     = float(np.corrcoef(proba_a, proba_b)[0, 1])

        report = {
            "n_samples":      len(raw_df),
            "agreement_rate": round(agreement_rate, 4),
            "proba_corr":     round(proba_corr, 4),
            "catboost_proba": proba_a.tolist(),
            "xgboost_proba":  proba_b.tolist(),
            "delong_test":    None,
            "mcnemar_test":   None,
            "catboost_metrics": {},
            "xgboost_metrics":  {},
        }

        if y_true is not None:
            y_true = np.array(y_true)

            # Metrikler
            for key, proba, pred in [
                ("catboost_metrics", proba_a, pred_a),
                ("xgboost_metrics",  proba_b, pred_b),
            ]:
                auc    = roc_auc_score(y_true, proba)
                lo, hi = _bootstrap_auc_ci(y_true, proba)
                report[key] = {
                    "roc_auc":      round(auc, 4),
                    "roc_auc_ci":   f"[{lo:.4f}, {hi:.4f}]",
                    "pr_auc":       round(average_precision_score(y_true, proba), 4),
                    "brier_score":  round(_brier_score(y_true, proba), 4),
                    "ece":          _expected_calibration_error(y_true, proba),
                }

            # DeLong testi
            try:
                report["delong_test"]  = delong_test(y_true, proba_a, proba_b)
            except Exception as exc:
                logger.error("DeLong testi hatası: %s", exc)

            # McNemar testi
            report["mcnemar_test"] = mcnemar_test(y_true, pred_a, pred_b)

            logger.info(
                "Champion ROC-AUC: %.4f %s | Challenger: %.4f %s",
                report["catboost_metrics"]["roc_auc"],
                report["catboost_metrics"]["roc_auc_ci"],
                report["xgboost_metrics"]["roc_auc"],
                report["xgboost_metrics"]["roc_auc_ci"],
            )
            if report["delong_test"]:
                d = report["delong_test"]
                logger.info(
                    "DeLong: Δ AUC = %.4f, z = %.4f, p = %.4f (%s)",
                    d["auc_diff"], d["z_stat"], d["p_value"],
                    "ANLAMLI" if d["significant_95"] else "anlamsız",
                )

        return report

    # -----------------------------------------------------------------------
    # Görselleştirme
    # -----------------------------------------------------------------------

    def plot_roc_comparison(self, report: dict) -> Optional[object]:
        """
        İki modelin olasılık dağılımını karşılaştıran violin + scatter grafiği.
        y_true olmadan çalışır.
        """
        if not _PLOTLY:
            logger.error("plotly yüklü değil.")
            return None

        proba_a = report["catboost_proba"]
        proba_b = report["xgboost_proba"]

        fig = go.Figure()
        fig.add_trace(go.Violin(
            y=proba_a, name="Champion (CatBoost)",
            box_visible=True, meanline_visible=True,
            fillcolor="#bfdbfe", line_color="#1d4ed8", opacity=0.7,
        ))
        fig.add_trace(go.Violin(
            y=proba_b, name="Challenger (XGBoost)",
            box_visible=True, meanline_visible=True,
            fillcolor="#fde68a", line_color="#d97706", opacity=0.7,
        ))
        fig.update_layout(
            title=dict(
                text="Champion / Challenger — Churn Olasılık Dağılımı",
                font_size=14, font_color="#0f172a",
            ),
            yaxis_title="Churn Olasılığı",
            height=380,
            paper_bgcolor="white",
            plot_bgcolor="#f8fafc",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=70, b=40, l=50, r=30),
        )
        return fig

    def plot_calibration_comparison(
        self,
        report: dict,
        y_true: np.ndarray,
    ) -> Optional[object]:
        """
        Kalibrasyon eğrisi (güvenilirlik diyagramı) karşılaştırması.
        İdeal model: 45 derece çizgisine yakın.
        """
        if not _PLOTLY:
            return None

        from sklearn.calibration import calibration_curve

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="#94a3b8"),
            name="Mükemmel Kalibrasyon",
        ))

        colors = [("#1d4ed8", "Champion (CatBoost)"), ("#d97706", "Challenger (XGBoost)")]
        for (color, label), proba_key in zip(
            colors, ["catboost_proba", "xgboost_proba"]
        ):
            proba_arr = np.array(report[proba_key])
            frac_pos, mean_pred = calibration_curve(y_true, proba_arr, n_bins=10)
            fig.add_trace(go.Scatter(
                x=mean_pred, y=frac_pos, mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=7),
                name=label,
            ))

        fig.update_layout(
            title=dict(text="Kalibrasyon Eğrisi (Güvenilirlik Diyagramı)", font_size=14),
            xaxis_title="Ortalama Tahmin Olasılığı",
            yaxis_title="Gerçek Churn Oranı",
            height=360,
            paper_bgcolor="white",
            plot_bgcolor="#f8fafc",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=70, b=40),
        )
        return fig

    def summary_table(self, report: dict) -> pd.DataFrame:
        """
        İki modelin tüm metriklerini tek bir karşılaştırma tablosuna döker.
        Tez tablosu için doğrudan kullanılabilir.
        """
        rows = []
        metric_labels = {
            "roc_auc":     "ROC-AUC",
            "roc_auc_ci":  "ROC-AUC 95% CI",
            "pr_auc":      "PR-AUC",
            "brier_score": "Brier Skoru ↓",
            "ece":         "ECE (Kalibrasyon Hatası) ↓",
        }
        for key, label in metric_labels.items():
            cat_val = report["catboost_metrics"].get(key, "—")
            xgb_val = report["xgboost_metrics"].get(key, "—")
            rows.append({
                "Metrik":                label,
                "Champion (CatBoost)":   cat_val,
                "Challenger (XGBoost)":  xgb_val,
            })

        rows.append({"Metrik": "Tahmin Uyuşma Oranı",
                     "Champion (CatBoost)": f"{report['agreement_rate']:.4f}",
                     "Challenger (XGBoost)": "—"})
        rows.append({"Metrik": "Olasılık Korelasyonu",
                     "Champion (CatBoost)": f"{report['proba_corr']:.4f}",
                     "Challenger (XGBoost)": "—"})

        if report.get("delong_test"):
            d = report["delong_test"]
            rows.append({
                "Metrik":               "DeLong p-değeri (H₀: AUC eşit)",
                "Champion (CatBoost)":  f"p={d['p_value']:.4f}",
                "Challenger (XGBoost)": "ANLAMLI" if d["significant_95"] else "Anlamsız",
            })
        if report.get("mcnemar_test"):
            m = report["mcnemar_test"]
            rows.append({
                "Metrik":               "McNemar p-değeri (H₀: hata dağılımı eşit)",
                "Champion (CatBoost)":  f"p={m['p_value']:.4f}",
                "Challenger (XGBoost)": "ANLAMLI" if m["significant_95"] else "Anlamsız",
            })

        return pd.DataFrame(rows)
