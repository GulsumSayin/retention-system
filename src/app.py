"""
app.py  —  Müşteri Tutundurma Zekâsı Platformu (Bitirme Tezi)
Streamlit tabanlı karar destek sistemi.

Çalıştırma:
    streamlit run src/app.py
"""

import json
import math
import os
import pickle
import sys
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# PATH
# ---------------------------------------------------------------------------
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
for _p in (_SRC_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Proje modülleri
# ---------------------------------------------------------------------------
from model_router         import ModelRouter
from agents               import RetentionAgent
from optimization         import RetentionOptimizer
from evaluation           import StrategyEvaluator
from shap_service         import ShapService
from data_validator       import DataValidator
from model_comparator     import ModelComparator
from sensitivity_analysis import run_oat_sensitivity, plot_tornado
from llm_service          import rule_based_portfolio_summary, rule_based_strategy_comment

# ===========================================================================
# Model metadata yükleme
# ===========================================================================
def _load_model_meta(model_key: str) -> dict:
    """artifacts/{model}/metadata.json dosyasından model bilgilerini okur."""
    path = os.path.join(_ROOT_DIR, "artifacts", model_key, "metadata.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _load_threshold(model_key: str) -> float | None:
    path = os.path.join(_ROOT_DIR, "artifacts", model_key, "threshold.pkl")
    try:
        with open(path, "rb") as f:
            return round(float(pickle.load(f)), 3)
    except Exception:
        return None

# ===========================================================================
# Yardımcı formatlama
# ===========================================================================
def _fmt_tl(x: float) -> str:
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def _fmt_pct(x: float) -> str:
    return f"%{x * 100:.1f}"

# ===========================================================================
# Özet sözlüğü
# ===========================================================================
def generate_summary(enriched: pd.DataFrame, optimized: pd.DataFrame,
                     candidate_pool: pd.DataFrame) -> dict:
    total      = len(enriched)
    avg_churn  = float(enriched["churn_proba"].mean())  if total > 0 else 0.0
    high_cnt   = int((enriched["risk_level"] == "Yüksek").sum()) if "risk_level" in enriched.columns else 0
    mid_cnt    = int((enriched["risk_level"] == "Orta").sum())
    low_cnt    = int((enriched["risk_level"] == "Düşük").sum())
    port_loss  = float(enriched["expected_loss"].sum())        if "expected_loss"        in enriched.columns else 0.0
    port_saved = float(enriched["expected_saved_value"].sum()) if "expected_saved_value" in enriched.columns else 0.0
    high_ratio = high_cnt / total if total > 0 else 0.0

    portfolio_risk = (
        "Yüksek Riskli Portföy"      if avg_churn >= 0.65 or high_ratio >= 0.60 else
        "Orta-Yüksek Riskli Portföy" if avg_churn >= 0.40 or high_ratio >= 0.30 else
        "Görece Dengeli Portföy"
    )
    urgency = (
        "Acil müdahale gerekli"   if high_ratio >= 0.60 and port_loss >= 5000 else
        "Öncelikli takip gerekli" if high_ratio >= 0.30 or  port_loss >= 2000 else
        "Rutin takip yeterli"
    )

    if len(optimized) > 0:
        sel_count  = len(optimized)
        sel_budget = float(optimized["offer_cost"].sum())
        sel_saved  = float(optimized["expected_saved_value"].sum())
        sel_net    = float(optimized["net_benefit"].sum())
        avg_roi    = float(optimized["roi"].mean())
        top_action = optimized["action_category"].value_counts().idxmax()
    else:
        sel_count = sel_budget = sel_saved = sel_net = avg_roi = 0
        top_action = "Tanımlanmadı"

    return dict(
        total_customers=total, candidate_count=len(candidate_pool),
        avg_churn=avg_churn, high_risk_count=high_cnt,
        medium_risk_count=mid_cnt, low_risk_count=low_cnt,
        portfolio_expected_loss=port_loss, portfolio_expected_saved=port_saved,
        portfolio_risk=portfolio_risk, urgency=urgency,
        selected_count=sel_count, selected_budget=sel_budget,
        selected_expected_saved=sel_saved, selected_net_benefit=sel_net,
        avg_roi=avg_roi, top_action=top_action,
    )

# ===========================================================================
# Grafik fonksiyonları
# ===========================================================================
def plot_risk_donut(risk_counts: pd.Series) -> go.Figure:
    color_map = {"Yüksek": "#ef4444", "Orta": "#f59e0b", "Düşük": "#22c55e"}
    fig = go.Figure(go.Pie(
        labels=risk_counts.index, values=risk_counts.values, hole=0.60,
        marker_colors=[color_map.get(l, "#94a3b8") for l in risk_counts.index],
        textinfo="label+percent", textfont_size=12,
        hovertemplate="%{label}: %{value} müşteri (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(t=48, b=20, l=10, r=10), height=270,
        paper_bgcolor="white",
        title=dict(text="Risk Dağılımı — Aday Havuzu", font_size=13,
                   font_color="#0f172a", x=0.5),
        annotations=[dict(text=f"<b>{risk_counts.sum()}</b><br><span style='font-size:11px'>aday</span>",
                          x=0.5, y=0.5, font_size=15, showarrow=False, font_color="#374151")],
    )
    return fig


def plot_action_bar(action_counts: pd.Series) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=action_counts.values, y=action_counts.index, orientation="h",
        marker=dict(color=action_counts.values,
                    colorscale=[[0, "#bfdbfe"], [1, "#1d4ed8"]], showscale=False),
        text=action_counts.values, textposition="outside",
        hovertemplate="%{y}: %{x} müşteri<extra></extra>",
    ))
    fig.update_layout(
        height=290, paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(t=48, b=10, l=10, r=50),
        title=dict(text="Aksiyon Dağılımı — Aday Havuzu", font_size=13,
                   font_color="#0f172a", x=0.5),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, tickfont_size=10),
    )
    return fig


def plot_clv_churn_scatter(df: pd.DataFrame) -> go.Figure:
    _med = df["estimated_clv"].median()
    fig = px.scatter(
        df, x="churn_proba", y="estimated_clv", color="risk_level",
        size="expected_loss", size_max=22,
        hover_data={"action_category": True, "MonthlyCharges": True,
                    "tenure": True, "churn_proba": ":.2f", "estimated_clv": ":,.0f"},
        color_discrete_map={"Yüksek": "#ef4444", "Orta": "#f59e0b", "Düşük": "#22c55e"},
        labels={"churn_proba": "Terk Riski", "estimated_clv": "Tahmini CLV (TL)",
                "risk_level": "Risk Seviyesi"},
        title="CLV × Churn Risk Matrisi — Aday Havuzu", opacity=0.82,
    )
    fig.add_vline(x=0.5, line_dash="dot", line_color="#94a3b8", line_width=1,
                  annotation_text="Eşik: 0.50", annotation_position="top right",
                  annotation_font_size=10, annotation_font_color="#64748b")
    fig.add_hline(y=_med, line_dash="dot", line_color="#94a3b8", line_width=1,
                  annotation_text=f"Medyan CLV: {_med:,.0f} TL",
                  annotation_position="right",
                  annotation_font_size=10, annotation_font_color="#64748b")
    fig.update_layout(
        height=390, paper_bgcolor="white", plot_bgcolor="#f8fafc",
        margin=dict(t=60, b=40, l=50, r=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, title_text=""),
        title=dict(font_size=13, font_color="#0f172a", x=0.0),
    )
    return fig


def plot_roi_distribution(optimized: pd.DataFrame) -> go.Figure:
    top    = optimized.nlargest(min(20, len(optimized)), "roi").reset_index(drop=True)
    colors = top["roi"].apply(
        lambda v: "#16a34a" if v >= 2.0 else "#2563eb" if v >= 1.0 else "#f59e0b"
    )
    fig = go.Figure(go.Bar(
        x=top.index + 1, y=top["roi"], marker_color=colors,
        text=top["roi"].apply(lambda v: f"{v:.1f}x"), textposition="outside",
        hovertemplate="Müşteri #%{x}<br>ROI: %{y:.2f}x<extra></extra>",
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="#94a3b8", line_width=1,
                  annotation_text="Kırılım noktası (ROI = 1x)",
                  annotation_position="right",
                  annotation_font_size=10, annotation_font_color="#64748b")
    fig.update_layout(
        height=310, paper_bgcolor="white", plot_bgcolor="white",
        title=dict(text="Seçilen Müşteri ROI Dağılımı (İlk 20)",
                   font_size=13, font_color="#0f172a", x=0.0),
        xaxis=dict(title="Müşteri Sırası (ROI'ye göre)", showgrid=False),
        yaxis=dict(title="ROI (net fayda / maliyet)", showgrid=True,
                   gridcolor="#f1f5f9", zeroline=True),
        margin=dict(t=60, b=50, l=60, r=30),
    )
    return fig


def plot_strategy_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    strategies   = comparison_df["strategy"].tolist()
    net_benefits = comparison_df["net_benefit"].tolist()
    rois         = comparison_df["avg_roi"].tolist()
    bar_colors   = ["#1d4ed8", "#64748b", "#94a3b8"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Net Fayda (TL)", x=strategies, y=net_benefits,
        marker_color=bar_colors,
        text=[f"{_fmt_tl(v)} TL" for v in net_benefits],
        textposition="outside", yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        name="Ort. ROI", x=strategies, y=rois,
        mode="lines+markers+text",
        marker=dict(size=10, color="#f59e0b"),
        line=dict(color="#f59e0b", width=2, dash="dot"),
        text=[f"{v:.2f}x" for v in rois], textposition="top center", yaxis="y2",
    ))
    fig.update_layout(
        height=380, paper_bgcolor="white", plot_bgcolor="white",
        title=dict(text="Strateji Karşılaştırması — Net Fayda & Ortalama ROI",
                   font_size=13, font_color="#0f172a"),
        yaxis=dict(title="Net Fayda (TL)", showgrid=True, gridcolor="#f1f5f9"),
        yaxis2=dict(title="Ort. ROI", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=70, b=50), barmode="group",
    )
    return fig

# ===========================================================================
# CSS — tam kontrol, temiz tasarım
# ===========================================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* --- Temel tipografi --- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    color: #0f172a !important;
}
.stApp { background: #f0f4f8 !important; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1300px; }

/* --- Sidebar --- */
section[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: none !important;
    min-width: 260px !important;
}
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
section[data-testid="stSidebar"] label { color: #cbd5e1 !important; font-size: 0.82rem !important; }
section[data-testid="stSidebar"] p { color: #94a3b8 !important; font-size: 0.82rem !important; }

/* --- Sidebar bölüm başlıkları --- */
.sb-header {
    font-size: 0.65rem;
    font-weight: 700;
    color: #475569 !important;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin: 18px 0 8px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e293b;
}

/* --- Sidebar bilgi kartı --- */
.sb-card {
    background: #1e293b;
    border-radius: 10px;
    padding: 12px 14px;
    margin: 8px 0 4px 0;
    border-left: 3px solid #2563eb;
}
.sb-card-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 5px;
    font-size: 0.78rem;
}
.sb-card-label { color: #64748b !important; }
.sb-card-value { color: #e2e8f0 !important; font-weight: 600; }
.sb-card-value.green { color: #4ade80 !important; }
.sb-card-value.yellow { color: #fbbf24 !important; }

/* --- Sidebar status badge --- */
.sb-status {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #15803d22;
    border: 1px solid #15803d;
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 0.78rem;
    color: #4ade80 !important;
    margin-top: 6px;
    font-weight: 500;
}
.sb-status.error {
    background: #991b1b22;
    border-color: #991b1b;
    color: #f87171 !important;
}
.sb-status.info {
    background: #1e40af22;
    border-color: #1e40af;
    color: #93c5fd !important;
}

/* --- Sidebar sistem notu --- */
.sb-note {
    font-size: 0.72rem;
    color: #475569 !important;
    line-height: 1.6;
    margin-top: 10px;
    padding: 10px 12px;
    background: #0c1929;
    border-radius: 8px;
}
.sb-note b { color: #64748b !important; }

/* --- Başlık alanı --- */
h1 {
    font-size: 1.75rem !important;
    font-weight: 800 !important;
    color: #0f172a !important;
    letter-spacing: -0.035em !important;
    line-height: 1.2 !important;
}
h2 { font-size: 1.2rem !important; font-weight: 700 !important; color: #1e293b !important; }
h3 { font-size: 1.0rem !important; font-weight: 600 !important; color: #334155 !important; }

/* --- Metodoloji rozet şeridi --- */
.method-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    margin: 10px 0 4px 0;
}
.method-badge {
    display: inline-block;
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    border-radius: 5px;
    padding: 3px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.method-badge.green { background:#f0fdf4; color:#15803d; border-color:#bbf7d0; }
.method-badge.amber { background:#fffbeb; color:#92400e; border-color:#fde68a; }
.method-badge.purple { background:#faf5ff; color:#6b21a8; border-color:#e9d5ff; }

/* --- Bölüm ayracı --- */
.section-divider {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 26px 0;
}

/* --- Özet paneller --- */
.summary-panel {
    background: white;
    border-radius: 14px;
    padding: 20px 22px;
    border-top: 4px solid #2563eb;
    box-shadow: 0 1px 8px rgba(0,0,0,0.05);
    margin-bottom: 12px;
}
.summary-panel.alert  { border-top-color: #ef4444; }
.summary-panel.neutral { border-top-color: #64748b; }
.summary-panel.success { border-top-color: #16a34a; }
.panel-title {
    font-size: 0.72rem;
    font-weight: 700;
    color: #64748b;
    margin-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: 0.09em;
}
.panel-body { font-size: 0.9rem; color: #374151; line-height: 1.9; }

/* --- Risk rozetleri --- */
.badge {
    display: inline-block;
    padding: 3px 11px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 0.78rem;
}
.badge-high   { background:#fef2f2; color:#b91c1c; border:1px solid #fecaca; }
.badge-mid    { background:#fffbeb; color:#92400e; border:1px solid #fde68a; }
.badge-low    { background:#f0fdf4; color:#166534; border:1px solid #bbf7d0; }
.badge-blue   { background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe; }

/* --- Müşteri analiz kartları --- */
.analysis-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    border-top: 3px solid #2563eb;
    margin-bottom: 2px;
}
.analysis-card.high-risk { border-top-color: #ef4444; }
.analysis-card.mid-risk  { border-top-color: #f59e0b; }
.card-meta { font-size: 0.68rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:4px; }
.card-title { font-size:0.95rem; font-weight:700; color:#0f172a; margin-bottom:8px; }
.card-body { font-size:0.85rem; color:#374151; line-height:1.75; }
.card-metrics {
    margin-top:12px; padding-top:9px; border-top:1px solid #f1f5f9;
    display:flex; gap:14px; flex-wrap:wrap; font-size:0.77rem; color:#64748b;
}
.card-metrics b { color:#374151; }

/* --- Metodoloji bilgi kutusu --- */
.method-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #2563eb;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 6px;
}
.method-box h4 { font-size:0.85rem; font-weight:700; color:#1e293b; margin:0 0 6px 0; }
.method-box p  { font-size:0.82rem; color:#475569; line-height:1.7; margin:0; }

/* --- Butonlar --- */
.stButton>button, .stDownloadButton>button {
    background: #1d4ed8 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.87rem !important;
    padding: 0.55rem 1.1rem !important;
    transition: background 0.15s !important;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    background: #1e40af !important;
}

/* --- Dosya yükleyici --- */
section[data-testid="stFileUploader"] {
    background: white !important;
    border: 2px dashed #cbd5e1 !important;
    border-radius: 12px !important;
    padding: 14px !important;
}

/* --- Metrik kartları --- */
[data-testid="stMetric"] {
    background: white !important;
    border-radius: 12px !important;
    padding: 14px 16px !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}
[data-testid="stMetricLabel"] { font-size: 0.78rem !important; color: #64748b !important; }
[data-testid="stMetricValue"] { font-size: 1.35rem !important; font-weight: 700 !important; }

/* --- DataFrame --- */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    border: 1px solid #e2e8f0 !important;
}
</style>
"""

# ===========================================================================
# Sayfa yapılandırması
# ===========================================================================
st.set_page_config(
    page_title="Müşteri Tutundurma Sistemi",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ===========================================================================
# Bileşen yükleme — hata yönetimiyle
# ===========================================================================
@st.cache_resource
def load_components():
    try:
        return (
            ModelRouter(),
            RetentionAgent(),
            RetentionOptimizer(),
            StrategyEvaluator(),
            ShapService(),
        )
    except FileNotFoundError as exc:
        st.error(
            f"Model artifact bulunamadı: {exc}\n\n"
            "`train/train_catboost.py` ve `train/train_xgboost.py` çalıştırılarak "
            "artifact klasörü oluşturulmalıdır."
        )
        st.stop()
    except Exception as exc:
        st.error(f"Bileşen yükleme hatası: {exc}")
        st.stop()

router, agent, optimizer, evaluator, shap_svc = load_components()

# ===========================================================================
# SIDEBAR
# ===========================================================================
with st.sidebar:

    # ── Logo / başlık ────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding:16px 4px 4px 4px">
        <div style="font-size:1.05rem;font-weight:800;color:#e2e8f0;letter-spacing:-0.02em;line-height:1.3">
            Tutundurma Zekâsı
        </div>
        <div style="font-size:0.72rem;color:#475569;margin-top:3px;letter-spacing:0.04em">
            IBM TELCO · ANALİTİK PLATFORM
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sistem Ayarları ──────────────────────────────────────────────────
    st.markdown('<div class="sb-header">Sistem Ayarları</div>', unsafe_allow_html=True)

    top_n           = st.slider("Tablo gösterim limiti", 5, 50, 10, 5,
                                help="Aday ve optimizasyon tablolarında gösterilecek satır sayısı")
    candidate_ratio = st.selectbox(
        "Aday havuzu büyüklüğü",
        [0.10, 0.20, 0.30], index=1,
        format_func=lambda x: f"En riskli %{int(x*100)} — öncelik skoru sırası",
        help="Portföyün yüksek-orta riskli dilimi; optimizasyon bu havuz üzerinde çalışır",
    )
    max_budget = st.number_input(
        "Kampanya bütçesi (TL)", min_value=0, value=2000, step=100,
        help="Greedy Knapsack algoritması bu kısıt altında ROI-maksimum müşteri kümesini seçer",
    )

    # ── Model Seçimi ─────────────────────────────────────────────────────
    st.markdown('<div class="sb-header">Tahmin Modeli</div>', unsafe_allow_html=True)

    model_choice = st.radio(
        "Aktif model",
        ["CatBoost  —  Champion", "XGBoost  —  Challenger"],
        index=0,
        help="Champion model varsayılan üretim modelidir. Challenger karşılaştırma için kullanılır.",
    )
    model_key = "catboost" if "CatBoost" in model_choice else "xgboost"

    # Model bilgi kartı
    _threshold = _load_threshold(model_key)
    _meta      = _load_model_meta(model_key)
    _roc       = _meta.get("metrics", {}).get("roc_auc")
    _prauc     = _meta.get("metrics", {}).get("pr_auc")
    _role      = "Champion" if model_key == "catboost" else "Challenger"
    _algo      = "CatBoostClassifier" if model_key == "catboost" else "XGBClassifier"

    _thr_str  = f"{_threshold:.3f}" if _threshold else "—"
    _roc_str  = f"{_roc:.4f}"       if _roc       else "—"
    _prauc_str = f"{_prauc:.4f}"    if _prauc      else "—"

    st.markdown(f"""
    <div class="sb-card">
        <div class="sb-card-row">
            <span class="sb-card-label">Algoritma</span>
            <span class="sb-card-value">{_algo}</span>
        </div>
        <div class="sb-card-row">
            <span class="sb-card-label">Rol</span>
            <span class="sb-card-value">{_role}</span>
        </div>
        <div class="sb-card-row">
            <span class="sb-card-label">Karar eşiği</span>
            <span class="sb-card-value yellow">{_thr_str}</span>
        </div>
        <div class="sb-card-row">
            <span class="sb-card-label">ROC-AUC</span>
            <span class="sb-card-value green">{_roc_str}</span>
        </div>
        <div class="sb-card-row" style="margin-bottom:0">
            <span class="sb-card-label">PR-AUC</span>
            <span class="sb-card-value green">{_prauc_str}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Dil Modeli ───────────────────────────────────────────────────────
    st.markdown('<div class="sb-header">Dil Modeli (Opsiyonel)</div>', unsafe_allow_html=True)

    enable_llm = st.checkbox(
        "Müşteri yorumu üret",
        value=False,
        help="Qwen 2.5 3B · llama.cpp · localhost:8080 — LLM kapalıyken tüm analizler çalışır",
    )
    llm_limit = st.slider("Yorum kartı sayısı", 2, 8, 6, 1, disabled=not enable_llm)

    if enable_llm:
        from llama_server_manager import is_server_running, start_server
        if not is_server_running():
            with st.spinner("Qwen 2.5 başlatılıyor..."):
                ok = start_server()
            if ok:
                st.markdown('<div class="sb-status">Dil modeli hazır</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="sb-status error">Sunucu başlatılamadı</div>', unsafe_allow_html=True)
                enable_llm = False
        else:
            st.markdown('<div class="sb-status">Dil modeli aktif — Qwen 2.5 3B</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="sb-status info">LLM devre dışı — kural tabanlı mod</div>', unsafe_allow_html=True)

    # ── Sistem notu ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="sb-note">
        <b>7 Katmanlı Mimari</b><br>
        Veri Doğrulama → Ön İşleme → Model Çıkarımı →
        Ajan Katmanı → Optimizasyon → Değerlendirme → Sunum<br><br>
        <b>Veri:</b> IBM Telco Customer Churn<br>
        <b>XAI:</b> SHAP TreeExplainer (Lundberg & Lee, 2017)<br>
        <b>Optimizasyon:</b> Greedy Knapsack (Dantzig, 1957)
    </div>
    """, unsafe_allow_html=True)

# ===========================================================================
# ANA SAYFA — Başlık
# ===========================================================================
st.markdown("""
<div style="margin-bottom:4px">
  <h1>Müşteri Tutundurma Zekâsı Platformu</h1>
  <p style="color:#64748b;font-size:0.9rem;margin:4px 0 8px 0;line-height:1.6">
    Churn riski tahmini &nbsp;·&nbsp; CLV analizi &nbsp;·&nbsp;
    Bütçe kısıtlı aksiyon optimizasyonu &nbsp;·&nbsp;
    SHAP açıklanabilirlik &nbsp;·&nbsp; Çok-strateji karşılaştırması
  </p>
  <div class="method-strip">
    <span class="method-badge">CatBoost Champion</span>
    <span class="method-badge">XGBoost Challenger</span>
    <span class="method-badge green">SHAP XAI</span>
    <span class="method-badge green">Leakage-Free</span>
    <span class="method-badge amber">Greedy Knapsack</span>
    <span class="method-badge amber">OAT Duyarlılık</span>
    <span class="method-badge purple">DeLong · McNemar</span>
    <span class="method-badge purple">Qwen 2.5 7B</span>
  </div>
</div>
<hr class="section-divider">
""", unsafe_allow_html=True)

# ===========================================================================
# ===========================================================================
# Dosya yükleme
# ===========================================================================
uploaded_file = st.file_uploader(
    "IBM Telco formatında müşteri verisi yükleyin (.csv)",
    type=["csv"],
    help="Zorunlu sütunlar: tenure, MonthlyCharges, TotalCharges, Contract, PaymentMethod, InternetService",
)
if uploaded_file is None:
    st.markdown("""
    <div class="summary-panel neutral" style="margin-top:12px">
      <div class="panel-title">Nasıl Kullanılır</div>
      <div class="panel-body">
        IBM Telco Customer Churn formatında bir CSV dosyası yükleyin ve
        <b>Analizi Başlat</b> düğmesine basın.<br>
        Sistem sırasıyla şu adımları çalıştırır:<br>
        <span class="badge badge-blue" style="margin:4px 2px">1 · Doğrulama</span>
        <span class="badge badge-blue" style="margin:4px 2px">2 · Ön İşleme</span>
        <span class="badge badge-blue" style="margin:4px 2px">3 · Tahmin</span>
        <span class="badge badge-blue" style="margin:4px 2px">4 · SHAP</span>
        <span class="badge badge-blue" style="margin:4px 2px">5 · Ajan</span>
        <span class="badge badge-blue" style="margin:4px 2px">6 · Optimizasyon</span>
        <span class="badge badge-blue" style="margin:4px 2px">7 · Değerlendirme</span>
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── CSV okuma ──────────────────────────────────────────────────────────────
try:
    raw_df = pd.read_csv(uploaded_file)
except Exception as _err:
    st.error(f"CSV okunamadı: {_err}  —  Dosyanın geçerli UTF-8 kodlamalı CSV olduğundan emin olun.")
    st.stop()

# ── Veri doğrulama — 6 katmanlı DataValidator ─────────────────────────────
_val_report = DataValidator().validate(raw_df)
if not _val_report.is_valid:
    st.error(_val_report.format_errors())
    st.stop()
if _val_report.warnings:
    with st.expander("Veri Kalitesi Uyarıları", expanded=True):
        for _w in _val_report.warnings:
            st.warning(_w)

_col_info, _col_btn = st.columns([3, 1])
with _col_info:
    st.markdown(
        f"Yüklenen veri: **{len(raw_df):,} müşteri** · **{len(raw_df.columns)} sütun** "
        f"· Churn oranı: **{raw_df['Churn'].map({'Yes':1,'No':0}).mean()*100:.1f}%**"
        if "Churn" in raw_df.columns else
        f"Yüklenen veri: **{len(raw_df):,} müşteri** · **{len(raw_df.columns)} sütun**"
    )

with st.expander("Veri Önizlemesi", expanded=False):
    st.dataframe(raw_df.head(8), use_container_width=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

if not st.button("  Analizi Başlat", use_container_width=True):
    st.stop()

# ===========================================================================
# Analiz pipeline
# ===========================================================================
_prog = st.progress(0, text="Başlatılıyor...")

# 1 — Model tahmini
_prog.progress(15, text="Model tahmini çalışıyor...")
try:
    predictions = router.predict(raw_df, model=model_key)
except Exception as _err:
    st.error(f"Tahmin hatası: {_err}  —  Artifact dosyalarının bütünlüğünü kontrol edin.")
    st.stop()

# 2 — SHAP fit
_prog.progress(35, text="SHAP açıklanabilirlik katmanı hazırlanıyor...")
try:
    _model_obj    = router.cat_service.model if model_key == "catboost" else router.xgb_service.model
    _meta_cols    = {"churn_proba", "predicted_churn", "threshold_used", "model_name"}
    _feature_cols = [c for c in predictions.columns if c not in _meta_cols]
    shap_svc.fit(_model_obj, predictions[_feature_cols])
    agent.set_shap_service(shap_svc, _feature_cols)
except Exception as _err:
    st.warning(f"SHAP başlatılamadı — kural tabanlı gerekçeler kullanılacak. ({_err})")

# 3 — Ajan katmanı
_prog.progress(55, text="Ajan katmanı çalışıyor — iş puanları, risk, aksiyon...")
enriched = agent.run(predictions)

# 4 — Aday havuzu + optimizasyon
_prog.progress(75, text="Greedy Knapsack optimizasyonu...")
cand_count     = min(max(10, math.ceil(len(enriched) * candidate_ratio)), len(enriched))
candidate_pool = enriched[enriched["risk_level"].isin(["Yüksek", "Orta"])].head(cand_count).copy()
if len(candidate_pool) == 0:
    st.warning("Aday havuzunda yüksek veya orta riskli müşteri bulunamadı.")
    st.stop()
optimized = optimizer.select_by_constraints(candidate_pool, max_budget=float(max_budget))
summary   = generate_summary(enriched, optimized, candidate_pool)

# 5 — Strateji değerlendirme
_prog.progress(90, text="Strateji karşılaştırması hesaplanıyor...")
comparison_df = evaluator.compare_all(optimized, candidate_pool, max_budget)
advantage     = evaluator.agent_advantage_summary(comparison_df)

_prog.progress(100, text="Tamamlandı.")
_prog.empty()

# ===========================================================================
# Portföy değerlendirmesi — kural tabanlı
# ===========================================================================
portfolio_comment = rule_based_portfolio_summary(summary)
st.markdown(f"""
<div class="summary-panel">
  <div class="panel-title">Portföy Değerlendirmesi</div>
  <div class="panel-body">{portfolio_comment}</div>
</div>""", unsafe_allow_html=True)

# ===========================================================================
# Yönetici özeti
# ===========================================================================
st.markdown("## Yönetici Özeti")
badge_cls = (
    "badge-high" if summary["avg_churn"] >= 0.60 else
    "badge-mid"  if summary["avg_churn"] >= 0.40 else
    "badge-low"
)
col_a, col_b = st.columns(2)
with col_a:
    st.markdown(f"""
    <div class="summary-panel">
      <div class="panel-title">Portföy Riski</div>
      <div class="panel-body">
        <span class="badge {badge_cls}">{summary['portfolio_risk']}</span><br><br>
        Toplam <b>{summary['total_customers']:,}</b> müşteri —
        <b style="color:#ef4444">{summary['high_risk_count']:,} yüksek</b>,
        <b style="color:#f59e0b">{summary['medium_risk_count']:,} orta</b>,
        <b style="color:#16a34a">{summary['low_risk_count']:,} düşük</b> riskli.<br>
        Ortalama terk riski: <b>{_fmt_pct(summary['avg_churn'])}</b>
        &nbsp;·&nbsp; Aciliyet: <b>{summary['urgency']}</b>
      </div>
    </div>""", unsafe_allow_html=True)
with col_b:
    st.markdown(f"""
    <div class="summary-panel">
      <div class="panel-title">Optimizasyon Çıktısı — {max_budget:,} TL Bütçe</div>
      <div class="panel-body">
        <b>{summary['selected_count']:,}</b> müşteri seçildi &nbsp;·&nbsp;
        Maliyet: <b>{_fmt_tl(summary['selected_budget'])} TL</b><br>
        Beklenen kurtarma: <b>{_fmt_tl(summary['selected_expected_saved'])} TL</b><br>
        Net fayda: <b>{_fmt_tl(summary['selected_net_benefit'])} TL</b>
        &nbsp;·&nbsp; Ort. ROI: <b>{summary['avg_roi']:.2f}x</b><br>
        En sık aksiyon kategorisi: <b>{summary['top_action']}</b>
      </div>
    </div>""", unsafe_allow_html=True)

# ===========================================================================
# KPI metrikleri — 8 kart
# ===========================================================================
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("## Portföy Metrikleri")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Toplam Müşteri",       f"{len(enriched):,}")
m2.metric("Ortalama Terk Riski", f"%{enriched['churn_proba'].mean()*100:.1f}",
          help="Tüm portföydeki müşterilerin ortalama terk etme eğilimi")
m3.metric("Aday Havuzu",          f"{len(candidate_pool):,}",
          help=f"Yüksek + Orta riskli, ilk %{int(candidate_ratio*100)}")
m4.metric("Bütçede Seçilen",      f"{len(optimized):,}",
          help="Greedy Knapsack ile bütçe kısıtına uyan müşteriler")

m5, m6, m7, m8 = st.columns(4)
m5.metric("Toplam Portföy CLV",   f"{_fmt_tl(enriched['estimated_clv'].sum())} TL",
          help="estimated_clv = LTV × (1 − p × 0.3)")
m6.metric("Portföy Beklenen Kayıp", f"{_fmt_tl(enriched['expected_loss'].sum())} TL",
          help="expected_loss = churn_proba × LTV")
m7.metric("Beklenen Kurtarma",
          f"{_fmt_tl(optimized['expected_saved_value'].sum()) if len(optimized)>0 else '0,00'} TL",
          help="Seçilen müşteriler için: expected_loss × retention_uplift")
m8.metric("Kullanılan Bütçe",
          f"{_fmt_tl(optimized['offer_cost'].sum()) if len(optimized)>0 else '0,00'} TL",
          help="Aksiyon maliyetleri ACTION_REGISTRY'den hesaplanır")

# ===========================================================================
# Görsel analizler
# ===========================================================================
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("## Görsel Analizler")
st.caption(
    f"Grafikler **aday havuzu** ({len(candidate_pool):,} müşteri, "
    f"yüksek + orta risk) üzerinden hesaplanmıştır. "
    f"Portföy geneli: {len(enriched):,} müşteri."
)

g1, g2 = st.columns(2)
with g1:
    st.plotly_chart(plot_risk_donut(candidate_pool["risk_level"].value_counts()),
                    use_container_width=True)
with g2:
    st.plotly_chart(plot_action_bar(candidate_pool["action_category"].value_counts()),
                    use_container_width=True)

st.plotly_chart(plot_clv_churn_scatter(candidate_pool), use_container_width=True)

if len(optimized) > 0:
    st.plotly_chart(plot_roi_distribution(optimized), use_container_width=True)

# ===========================================================================
# SHAP açıklanabilirlik
# ===========================================================================
if shap_svc._fitted:
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("## Müşteri Kaybını Etkileyen Faktörler")
    st.caption(
        "Bu grafik, müşterilerin terk etme kararını en çok etkileyen faktörleri gösterir — "
        "hangisine ne kadar önem verildiğini somut olarak ortaya koyar. "
        "Teknik not: Ortalama mutlak SHAP değeri · TreeExplainer (Lundberg & Lee, NeurIPS 2017) · "
        "aday havuzu üzerinde hesaplanmıştır."
    )

    _non_feature = {
        "churn_proba","predicted_churn","threshold_used","model_name",
        "risk_level","action_category","action_detail","action_channel",
        "action_priority","personalization_note","action","action_reason",
        "retention_uplift","expected_saved_value","estimated_clv",
        "expected_loss","priority_score","ranking_score","lifetime_value",
    }
    _shap_cols = [c for c in candidate_pool.columns if c not in _non_feature]

    shap_fig = shap_svc.plot_shap_bar(
        candidate_pool[_shap_cols], top_n=10,
        title="Churn Tahminine En Çok Katkıda Bulunan 10 Değişken",
    )
    if shap_fig:
        st.plotly_chart(shap_fig, use_container_width=True)

    with st.expander("Terk Riskini Etkileyen Faktörler — Detaylı Sıralama (İlk 15)", expanded=False):
        st.caption("Değer ne kadar yüksekse, o faktör müşteri kaybını tahmin etmede o kadar belirleyicidir.")
        _imp_df = shap_svc.global_importance(candidate_pool[_shap_cols], top_n=15)
        if not _imp_df.empty:
            st.dataframe(
                _imp_df[["label","mean_abs_shap"]].rename(
                    columns={"label":"Faktör","mean_abs_shap":"Belirleyicilik Skoru"}
                ),
                use_container_width=True,
            )

# ===========================================================================
# Aday müşteri tablosu
# ===========================================================================
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown(f"## Aday Müşteri Listesi — İlk {min(top_n, len(candidate_pool))}")
st.caption(
    "Öncelik skoru = expected_loss × (1 + IsMonthToMonth) × (1 + ServiceIntensity) "
    "— yüksekten düşüğe sıralı"
)

CAND_COLS = [
    "model_name","tenure","MonthlyCharges","estimated_clv","churn_proba",
    "expected_loss","priority_score","risk_level","action_category",
    "action_detail","action_channel","action_reason","personalization_note",
]
CAND_RENAME = {
    "model_name":"Model","tenure":"Süre (ay)","MonthlyCharges":"Aylık Ücret (TL)",
    "estimated_clv":"Tahmini CLV","churn_proba":"Terk Riski",
    "expected_loss":"Beklenen Kayıp","priority_score":"Öncelik Skoru",
    "risk_level":"Risk","action_category":"Aksiyon Kategorisi",
    "action_detail":"Önerilen Aksiyon","action_channel":"Kanal",
    "action_reason":"Neden Risk Altında?","personalization_note":"Temsilci Notu",
}
_show = [c for c in CAND_COLS if c in candidate_pool.columns]
st.dataframe(
    candidate_pool[_show].head(top_n).rename(columns=CAND_RENAME),
    use_container_width=True,
)

# ===========================================================================
# Optimizasyon sonucu
# ===========================================================================
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown(f"## Optimizasyon Sonucu — {max_budget:,} TL Bütçe")

if len(optimized) == 0:
    st.warning(
        "Belirlenen bütçe dahilinde pozitif net faydası olan müşteri seçilemedi. "
        "Bütçeyi artırmayı veya aday havuzu oranını genişletmeyi deneyin."
    )
else:
    optimized_display = optimized.head(top_n).copy()

    # LLM yorum kartları
    if enable_llm:
        with st.spinner("Müşteri analizleri hazırlanıyor (Qwen 2.5)..."):
            from llm_service import LLMService
            if "llm_svc" not in st.session_state:
                st.session_state["llm_svc"] = LLMService()
            optimized_display = st.session_state["llm_svc"].add_llm_comment(
                optimized_display, limit=llm_limit
            )

        st.markdown("### Müşteri Bazlı Bağlamsal Analizler")
        st.caption("SHAP tabanlı gerekçe · Qwen 2.5 3B yorumu · Müşteri temsilcisi odaklı")

        for i in range(0, min(llm_limit, len(optimized_display)), 2):
            pair = optimized_display.iloc[i:i+2]
            cols = st.columns(2)
            for col, (_, row) in zip(cols, pair.iterrows()):
                risk       = row.get("risk_level", "")
                risk_badge = {
                    "Yüksek": '<span class="badge badge-high">Yüksek Risk</span>',
                    "Orta":   '<span class="badge badge-mid">Orta Risk</span>',
                    "Düşük":  '<span class="badge badge-low">Düşük Risk</span>',
                }.get(risk, "")
                card_cls = "analysis-card " + (
                    "high-risk" if risk == "Yüksek" else
                    "mid-risk"  if risk == "Orta"   else ""
                )
                comment = (row.get("llm_comment") or
                           "<em style='color:#9ca3af'>Analiz üretilemedi — LLM bağlantısı yok.</em>")
                with col:
                    st.markdown(f"""
                    <div class="{card_cls}">
                      <div class="card-meta">{row.get('model_name','')} &nbsp;·&nbsp; {row.get('action_channel','')}</div>
                      <div class="card-title">{row.get('action_detail','')}</div>
                      {risk_badge}<br><br>
                      <div class="card-body">{comment}</div>
                      <div class="card-metrics">
                        <span>Terk Riski <b>%{row.get('churn_proba',0)*100:.1f}</b></span>
                        <span>CLV <b>{row.get('estimated_clv',0):,.0f} TL</b></span>
                        <span>Geri Dönüş <b>{row.get('roi',0):.1f}x</b></span>
                        <span>Net Kazanç <b>{row.get('net_benefit',0):,.0f} TL</b></span>
                      </div>
                    </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

    # Optimizasyon tablosu
    OPT_COLS = [
        "model_name","MonthlyCharges","estimated_clv","churn_proba",
        "expected_loss","expected_saved_value","offer_cost","net_benefit",
        "roi","risk_level","action_category","action_detail","action_channel","action_reason",
    ]
    OPT_RENAME = {
        "model_name":"Model","MonthlyCharges":"Aylık Ücret",
        "estimated_clv":"CLV","churn_proba":"Terk Riski",
        "expected_loss":"Beklenen Kayıp","expected_saved_value":"Beklenen Kurtarma",
        "offer_cost":"Aksiyon Maliyeti","net_benefit":"Net Fayda",
        "roi":"ROI","risk_level":"Risk","action_category":"Aksiyon Kategorisi",
        "action_detail":"Önerilen Aksiyon","action_channel":"Kanal",
        "action_reason":"Neden Risk Altında?",
    }
    _opt_show = [c for c in OPT_COLS if c in optimized_display.columns]
    st.dataframe(
        optimized_display[_opt_show].rename(columns=OPT_RENAME),
        use_container_width=True,
    )

# ===========================================================================
# Strateji karşılaştırması
# ===========================================================================
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("## Strateji Karşılaştırması")
st.caption(
    "Önerilen kişiselleştirilmiş sistem vs. iki naif alt sınır — "
    "tüm stratejiler greedy net_benefit sıralaması ile bütçe kısıtına tabi tutulmuştur "
    "(karşılaştırma adaleti)."
)

st.plotly_chart(plot_strategy_comparison(comparison_df), use_container_width=True)

comp_display = comparison_df.copy()
comp_display["total_cost"]      = comp_display["total_cost"].apply(lambda x: f"{_fmt_tl(x)} TL")
comp_display["expected_saved"]  = comp_display["expected_saved"].apply(lambda x: f"{_fmt_tl(x)} TL")
comp_display["net_benefit"]     = comp_display["net_benefit"].apply(lambda x: f"{_fmt_tl(x)} TL")
comp_display["avg_roi"]         = comp_display["avg_roi"].apply(lambda x: f"{x:.2f}x")
comp_display["cost_efficiency"] = comp_display["cost_efficiency"].apply(lambda x: f"{x:.2f}x")
comp_display["precision_at_k"]  = comp_display["precision_at_k"].apply(
    lambda x: f"%{x*100:.1f}" if x is not None else "—"
)
comp_display.columns = [
    "Strateji","Seçilen","Maliyet","Beklenen Kurtarma",
    "Net Fayda","Ort. ROI","Maliyet Verimliliği","Precision@K",
]
st.dataframe(comp_display.set_index("Strateji"), use_container_width=True)

if advantage["agent_is_best"]:
    st.success(
        f"Yapay Zekâ Destekli Akıllı Yaklaşım, Geleneksel Toplu Yaklaşım'a göre "
        f"**%{advantage['vs_baseline_pct']:.1f}**, Risk Bazlı Sabit Aksiyon'a göre "
        f"**%{advantage['vs_risk_only_pct']:.1f}** daha yüksek net kazanç sağlamaktadır."
    )
else:
    st.info(
        "Mevcut bütçe veya aday havuzu büyüklüğü kısıtları nedeniyle Yapay Zekâ Destekli "
        "Akıllı Yaklaşım en yüksek net kazancı sağlayamamıştır. "
        "Kampanya bütçesini artırın veya daha geniş bir müşteri havuzu seçin."
    )

strategy_comment = rule_based_strategy_comment(comparison_df, advantage)
st.markdown(f"""
<div class="summary-panel {'success' if advantage['agent_is_best'] else 'neutral'}"
     style="margin-top:10px">
  <div class="panel-title">Strateji Değerlendirmesi</div>
  <div class="panel-body">{strategy_comment}</div>
</div>""", unsafe_allow_html=True)

# ===========================================================================
# Champion / Challenger model karşılaştırması
# ===========================================================================
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
with st.expander("Model Karşılaştırması — Champion / Challenger", expanded=False):
    st.caption(
        "CatBoost (Champion) ve XGBoost (Challenger) churn olasılık dağılımları · "
        "Gerçek etiket olmadan istatistiksel testler (DeLong, McNemar) çalıştırılamaz; "
        "test verisi sağlandığında model_comparator.py'de eksiksiz uygulanmıştır."
    )
    try:
        with st.spinner("İki model karşılaştırılıyor..."):
            _comp_rpt = ModelComparator(router).compare(raw_df, y_true=None)

        _cv1, _cv2 = st.columns(2)
        _cv1.metric("Tahmin Uyuşma Oranı", f"%{_comp_rpt['agreement_rate']*100:.1f}",
                    help="İki modelin aynı kararı ürettiği müşteri oranı")
        _cv2.metric("Olasılık Korelasyonu", f"{_comp_rpt['proba_corr']:.3f}",
                    help="churn_proba değerleri arasındaki Pearson korelasyonu")

        _vio = ModelComparator(router).plot_roc_comparison(_comp_rpt)
        if _vio:
            st.plotly_chart(_vio, use_container_width=True)

        st.markdown("""
        <div class="method-box">
            <h4>Uygulanabilir İstatistiksel Testler (Etiketli Veri Gerektirir)</h4>
            <p>
            <b>DeLong AUC Testi</b> — H₀: AUC(CatBoost) = AUC(XGBoost)
            (DeLong et al., Biometrics 44(3), 1988)<br>
            <b>McNemar Testi</b> — H₀: İki modelin hata dağılımı aynı
            (McNemar 1947, süreklilik düzeltmesi Edwards 1948)<br>
            <b>Bootstrap 95% CI</b> — 1000 iterasyon, stratified resampling
            </p>
        </div>
        """, unsafe_allow_html=True)
    except Exception as _err:
        st.error(f"Model karşılaştırma hatası: {_err}")

# ===========================================================================
# CLV Katsayıları Duyarlılık Analizi
# ===========================================================================
with st.expander("Duyarlılık Analizi — CLV Katsayıları (OAT)", expanded=False):
    st.caption(
        "Her CLV katsayısı sırayla ±10%, ±20%, ±30% değiştirilir; diğerleri nominal tutulur. "
        "En geniş tornadogram barı = en kritik katsayı = öncelikli A/B testi hedefi."
    )
    try:
        with st.spinner("OAT duyarlılık analizi çalışıyor..."):
            _cand_for_sens = enriched[enriched["risk_level"].isin(["Yüksek","Orta"])].copy()
            _sens_df = run_oat_sensitivity(_cand_for_sens, max_budget=float(max_budget))

        _fig_tor = plot_tornado(_sens_df)
        if _fig_tor:
            st.plotly_chart(_fig_tor, use_container_width=True)

        _tbl = (
            _sens_df.groupby("label")["net_benefit_change_pct"]
            .apply(lambda x: x.abs().max())
            .reset_index()
            .rename(columns={"label":"CLV Katsayısı","net_benefit_change_pct":"Maks. Net Fayda Değişimi (%)"})
            .sort_values("Maks. Net Fayda Değişimi (%)", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(_tbl, use_container_width=True)

        st.markdown("""
        <div class="method-box">
            <h4>Yorum ve Öneriler</h4>
            <p>Sonuçların büyük bölümünde düşük değişim gözlemleniyorsa mevcut
            katsayı varsayımları yeterince sağlamdır ve kampanya planlaması
            güvenle sürdürülebilir. Belirli bir faktörde yüksek değişkenlik
            görülüyorsa o faktöre ilişkin gerçek kampanya verisi öncelikli
            olarak toplanmalı ve model güncellenmelidir.</p>
        </div>
        """, unsafe_allow_html=True)
    except Exception as _err:
        st.error(f"Duyarlılık analizi hatası: {_err}")

# ===========================================================================
# CSV çıktıları — timestamped alt klasör
# ===========================================================================
_run_ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
outputs_dir = os.path.join(_ROOT_DIR, "outputs", _run_ts)
os.makedirs(outputs_dir, exist_ok=True)

enriched.to_csv(      os.path.join(outputs_dir, "tum_sonuclar.csv"),           index=False)
candidate_pool.to_csv(os.path.join(outputs_dir, "aday_havuzu.csv"),            index=False)
optimized.to_csv(     os.path.join(outputs_dir, "optimizasyon.csv"),           index=False)
comparison_df.to_csv( os.path.join(outputs_dir, "strateji_karsilastirma.csv"), index=False)

if shap_svc._fitted and _shap_cols:
    _imp_exp = shap_svc.global_importance(candidate_pool[_shap_cols], top_n=20)
    if not _imp_exp.empty:
        _imp_exp.to_csv(os.path.join(outputs_dir, "shap_feature_importance.csv"), index=False)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("## Çıktıları İndir")
st.caption(f"Çalıştırma klasörü: `outputs/{_run_ts}/`")

_dl_cols = st.columns(5)
_downloads = [
    ("Tüm Sonuçlar",          "tum_sonuclar.csv"),
    ("Aday Havuzu",           "aday_havuzu.csv"),
    ("Optimizasyon",          "optimizasyon.csv"),
    ("Strateji Karşılaştırma","strateji_karsilastirma.csv"),
    ("SHAP Önemi",            "shap_feature_importance.csv"),
]
for col, (label, fname) in zip(_dl_cols, _downloads):
    fpath = os.path.join(outputs_dir, fname)
    with col:
        if os.path.exists(fpath):
            with open(fpath, "rb") as f:
                st.download_button(label=label, data=f, file_name=fname,
                                   mime="text/csv", use_container_width=True)
