"""
sensitivity_analysis.py
CLV Katsayıları Duyarlılık Analizi (What-If Analizi)

Akademik Motivasyon:
  agents.py'deki CLV formülü:
    LTV = MonthlyCharges × (6 + tenure × α) × contract_mult

  Buradaki katsayılar (α=0.5, contract_mult={1.0, 1.3, 1.6}, iskonto=0.3)
  iş bilgisine dayalı varsayımsaldır. Jüri "bu katsayıları nereden buldunuz?"
  diye sorduğunda duyarlılık analizi yanıtı sağlar:
  "Katsayıları ±%30 değiştirdiğimizde optimizasyon çıktısı nasıl değişiyor?"

  Bu analiz iki soruyu yanıtlar:
  1. Model çıktıları katsayı seçimine ne kadar duyarlı?
     (Düşük duyarlılık → varsayımların kalibrasyona ihtiyacı az)
  2. Hangi katsayı en kritik? (Önceliklendirilmesi gereken validasyon hedefi)

Yöntem:
  Tek değişkenli (OAT — One-At-a-Time) duyarlılık analizi:
    - Her katsayı sırayla ±%10, ±%20, ±%30 oranında değiştirilir.
    - Diğer katsayılar nominal değerlerinde tutulur.
    - Her senaryo için toplam net fayda ve seçilen müşteri sayısı hesaplanır.
    - Sonuçlar tornadogram ile görselleştirilir.

Tez Bölümü:
  "5.3 CLV Katsayıları Duyarlılık Analizi" — sonuçlar bu modülden üretilir.
"""

import logging
from dataclasses import dataclass
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
# Nominal Katsayılar (agents.py ile tutarlı)
# ===========================================================================

@dataclass
class CLVParams:
    """CLV formülü katsayıları."""
    tenure_multiplier:      float = 0.5    # α: LTV = MC × (6 + tenure × α)
    base_horizon:           float = 6.0    # sabit ay ufku
    contract_mult_mtm:      float = 1.0    # Month-to-month çarpanı
    contract_mult_one_year: float = 1.3    # One year çarpanı
    contract_mult_two_year: float = 1.6    # Two year çarpanı
    churn_discount:         float = 0.3    # iskonto katsayısı (estimated_clv için)


NOMINAL = CLVParams()


# ===========================================================================
# Senaryo Hesaplama
# ===========================================================================

def compute_net_benefit_for_params(
    df: pd.DataFrame,
    params: CLVParams,
    max_budget: float = 2000.0,
) -> dict:
    """
    Verilen CLV katsayılarıyla tek bir senaryo için toplam net fayda ve
    seçilen müşteri sayısını hesaplar.

    Parametreler
    ------------
    df         : RetentionAgent çıktısı (churn_proba, action, offer_cost sütunları içerir)
    params     : CLV katsayıları
    max_budget : Kampanya bütçesi

    Döndürür
    --------
    dict: net_benefit_total, selected_count, total_cost
    """
    result = df.copy()

    contract_mult = pd.Series(1.0, index=result.index)
    if "Contract" in result.columns:
        contract_mult = result["Contract"].map({
            "Month-to-month": params.contract_mult_mtm,
            "One year":       params.contract_mult_one_year,
            "Two year":       params.contract_mult_two_year,
        }).fillna(1.0)

    lifetime_value = (
        result["MonthlyCharges"]
        * (params.base_horizon + result.get("tenure", 0) * params.tenure_multiplier)
        * contract_mult
    )
    expected_loss = result["churn_proba"] * lifetime_value

    if "retention_uplift" not in result.columns:
        result["retention_uplift"] = 0.15

    result["expected_saved_value"] = expected_loss * result["retention_uplift"]

    if "offer_cost" not in result.columns:
        result["offer_cost"] = 15.0

    result["net_benefit"] = result["expected_saved_value"] - result["offer_cost"]

    # Bütçe kısıtı + pozitif filtre
    eligible = result[
        (result["net_benefit"] > 0) &
        (result.get("action", "") != "müdahale gerekmiyor")
    ].copy()
    eligible = eligible.sort_values("net_benefit", ascending=False)

    selected, total_cost = [], 0.0
    for _, row in eligible.iterrows():
        if total_cost + row["offer_cost"] <= max_budget:
            selected.append(row)
            total_cost += row["offer_cost"]

    if not selected:
        return {"net_benefit_total": 0.0, "selected_count": 0, "total_cost": 0.0}

    sel_df = pd.DataFrame(selected)
    return {
        "net_benefit_total": round(float(sel_df["net_benefit"].sum()), 2),
        "selected_count":    len(sel_df),
        "total_cost":        round(total_cost, 2),
    }


# ===========================================================================
# OAT Duyarlılık Analizi
# ===========================================================================

_PARAM_META: dict[str, dict] = {
    "tenure_multiplier": {
        "label":   "Tenure Çarpanı (α)",
        "nominal": NOMINAL.tenure_multiplier,
        "unit":    "oran",
    },
    "base_horizon": {
        "label":   "Temel Ay Ufku (sabit)",
        "nominal": NOMINAL.base_horizon,
        "unit":    "ay",
    },
    "contract_mult_one_year": {
        "label":   "1 Yıllık Sözleşme Çarpanı",
        "nominal": NOMINAL.contract_mult_one_year,
        "unit":    "oran",
    },
    "contract_mult_two_year": {
        "label":   "2 Yıllık Sözleşme Çarpanı",
        "nominal": NOMINAL.contract_mult_two_year,
        "unit":    "oran",
    },
    "churn_discount": {
        "label":   "Churn İskonto Katsayısı",
        "nominal": NOMINAL.churn_discount,
        "unit":    "oran",
    },
}


def run_oat_sensitivity(
    df: pd.DataFrame,
    perturbations: list[float] | None = None,
    max_budget: float = 2000.0,
) -> pd.DataFrame:
    """
    One-At-a-Time (OAT) duyarlılık analizi.

    Her katsayı sırayla perturbations oranlarında değiştirilir,
    diğerleri nominal değerinde tutulur.

    Döndürür
    --------
    pd.DataFrame sütunları:
      parameter, label, perturbation_pct, param_value,
      net_benefit_total, selected_count, net_benefit_change_pct
    """
    if perturbations is None:
        perturbations = [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30]

    # Nominal senaryo
    nominal_result = compute_net_benefit_for_params(df, NOMINAL, max_budget)
    nominal_nb     = nominal_result["net_benefit_total"]
    logger.info("Nominal net fayda: %.2f TL (%d müşteri)", nominal_nb, nominal_result["selected_count"])

    rows = []
    for param_name, meta in _PARAM_META.items():
        for pert in perturbations:
            new_val = meta["nominal"] * (1 + pert)
            test_params = CLVParams(
                tenure_multiplier      = NOMINAL.tenure_multiplier,
                base_horizon           = NOMINAL.base_horizon,
                contract_mult_mtm      = NOMINAL.contract_mult_mtm,
                contract_mult_one_year = NOMINAL.contract_mult_one_year,
                contract_mult_two_year = NOMINAL.contract_mult_two_year,
                churn_discount         = NOMINAL.churn_discount,
            )
            setattr(test_params, param_name, new_val)
            result = compute_net_benefit_for_params(df, test_params, max_budget)

            change_pct = (
                (result["net_benefit_total"] - nominal_nb) / abs(nominal_nb) * 100
                if nominal_nb != 0 else 0.0
            )
            rows.append({
                "parameter":           param_name,
                "label":               meta["label"],
                "perturbation_pct":    round(pert * 100, 0),
                "param_value":         round(new_val, 4),
                "net_benefit_total":   result["net_benefit_total"],
                "selected_count":      result["selected_count"],
                "net_benefit_change_pct": round(change_pct, 2),
            })

    return pd.DataFrame(rows)


# ===========================================================================
# Görselleştirme
# ===========================================================================

def plot_tornado(sensitivity_df: pd.DataFrame) -> Optional[object]:
    """
    Tornadogram: en etkili katsayıyı en üstte gösterir.
    Net fayda değişimi (%) ±%30 pertürbasyon için barlar.

    Akademik yorum: En geniş bar = en kritik katsayı = öncelikli validasyon hedefi.
    """
    if not _PLOTLY:
        logger.error("plotly yüklü değil.")
        return None

    # Sadece ±%30 pertürbasyonlar
    df30 = sensitivity_df[sensitivity_df["perturbation_pct"].abs() == 30].copy()

    pivot = df30.pivot_table(
        index="label",
        columns="perturbation_pct",
        values="net_benefit_change_pct",
    ).reset_index()

    # Toplam salınım genişliğine göre sırala
    pivot["range"] = pivot.get(30.0, 0) - pivot.get(-30.0, 0)
    pivot = pivot.sort_values("range", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=pivot["label"],
        x=pivot.get(-30.0, 0),
        name="-30% Değişim",
        orientation="h",
        marker_color="#ef4444",
        hovertemplate="%{y}<br>-30%%: %{x:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=pivot["label"],
        x=pivot.get(30.0, 0),
        name="+30% Değişim",
        orientation="h",
        marker_color="#22c55e",
        hovertemplate="%{y}<br>+30%%: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        barmode="overlay",
        title=dict(
            text="CLV Katsayıları Duyarlılık Analizi (Tornadogram)",
            font_size=14, font_color="#0f172a",
        ),
        xaxis=dict(title="Net Fayda Değişimi (%)", showgrid=True, gridcolor="#f1f5f9", zeroline=True),
        yaxis=dict(showgrid=False, tickfont_size=11),
        height=380,
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=70, b=40, l=10, r=30),
    )
    return fig


def plot_sensitivity_lines(sensitivity_df: pd.DataFrame) -> Optional[object]:
    """
    Her katsayı için pertürbasyon (%) vs net fayda değişimi (%) çizgi grafiği.
    """
    if not _PLOTLY:
        return None

    fig    = go.Figure()
    colors = ["#1d4ed8", "#ef4444", "#22c55e", "#f59e0b", "#8b5cf6"]
    params = sensitivity_df["label"].unique()

    for color, param_label in zip(colors, params):
        sub = sensitivity_df[sensitivity_df["label"] == param_label].sort_values("perturbation_pct")
        fig.add_trace(go.Scatter(
            x=sub["perturbation_pct"],
            y=sub["net_benefit_change_pct"],
            mode="lines+markers",
            name=param_label,
            line=dict(color=color, width=2),
            marker=dict(size=6),
            hovertemplate=f"{param_label}<br>Değişim: %{{x:.0f}}%<br>Net Fayda: %{{y:.1f}}%<extra></extra>",
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8", line_width=1)
    fig.update_layout(
        title=dict(text="CLV Katsayıları — Pertürbasyon vs Net Fayda Değişimi", font_size=14),
        xaxis=dict(title="Katsayı Değişimi (%)", showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(title="Net Fayda Değişimi (%)", showgrid=True, gridcolor="#f1f5f9", zeroline=True),
        height=380,
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font_size=10),
        margin=dict(t=70, b=40),
    )
    return fig


def sensitivity_summary_table(sensitivity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Her katsayı için ±%30 pertürbasyondaki maksimum sapma ve yön.
    Tez tablosu için hazır format.
    """
    rows = []
    for label in sensitivity_df["label"].unique():
        sub       = sensitivity_df[sensitivity_df["label"] == label]
        max_up    = sub[sub["perturbation_pct"] > 0]["net_benefit_change_pct"].max()
        max_down  = sub[sub["perturbation_pct"] < 0]["net_benefit_change_pct"].min()
        magnitude = max(abs(max_up), abs(max_down)) if not (np.isnan(max_up) or np.isnan(max_down)) else 0

        rows.append({
            "Katsayı":                    label,
            "+30% Etkisi (%)":            round(max_up, 2),
            "-30% Etkisi (%)":            round(max_down, 2),
            "Maksimum Sapma (% mutlak)":  round(magnitude, 2),
            "Kritiklik":                  "Yüksek" if magnitude > 15 else ("Orta" if magnitude > 5 else "Düşük"),
        })

    return pd.DataFrame(rows).sort_values("Maksimum Sapma (% mutlak)", ascending=False)
