"""
flask_app.py — Müşteri Tutundurma Zekâsı Platformu (Flask)

Mimari:
  Bu dosya yalnızca HTTP katmanını yönetir. İş mantığının tamamı src/ içindeki
  modüllerde kalır; Flask view'ları yalnızca serialize / deserialize görevi görür.
  Bu ayrım, Single Responsibility Principle'ı (SRP) uygular.

Çalıştırma:
    python flask_app.py
    Tarayıcı: http://localhost:5000

Üretim:
    gunicorn -w 2 flask_app:app
"""

import io
import json
import logging
import math
import os
import pickle
import sys
import uuid
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from flask import Flask, jsonify, render_template, request, session

# ---------------------------------------------------------------------------
# Path kurulumu — src/ modülleri erişilebilir hale getir
# ---------------------------------------------------------------------------
_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR  = os.path.join(_ROOT_DIR, "src")
for _p in (_SRC_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Proje modülleri (iş mantığı — değiştirilmeden kullanılır)
# ---------------------------------------------------------------------------
from model_router         import ModelRouter
from agents               import RetentionAgent
from optimization         import RetentionOptimizer
from evaluation           import StrategyEvaluator, ABTestSimulator
from shap_service         import ShapService
from data_validator       import DataValidator
from model_comparator     import ModelComparator
from sensitivity_analysis import run_oat_sensitivity
from llm_service          import (
    rule_based_portfolio_summary,
    rule_based_strategy_comment,
    LLMService,
    LLMGuardrail,
)

# ---------------------------------------------------------------------------
# Uygulama konfigürasyonu
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB dosya limiti
app.config["JSON_SORT_KEYS"]     = False               # Sütun sırası korunsun
app.secret_key = os.environ.get("SECRET_KEY", "retention-platform-secret-2025")

# ---------------------------------------------------------------------------
# Bileşen havuzu — uygulama ömrü boyunca tek örnek (Singleton pattern)
# ---------------------------------------------------------------------------
_COMPONENTS: dict = {}

def _get_components() -> dict:
    """Model bileşenlerini ilk çağrıda yükler ve önbellekte tutar."""
    if not _COMPONENTS:
        logger.info("Bileşenler yükleniyor...")
        _COMPONENTS["router"]    = ModelRouter()
        _COMPONENTS["agent"]     = RetentionAgent()
        _COMPONENTS["optimizer"] = RetentionOptimizer()
        _COMPONENTS["evaluator"] = StrategyEvaluator()
        _COMPONENTS["shap_svc"]  = ShapService()
        _COMPONENTS["ab_sim"]    = ABTestSimulator()
        logger.info("Tüm bileşenler hazır.")
    return _COMPONENTS

# ---------------------------------------------------------------------------
# Kalıcılık (Persistence) yardımcıları
# ---------------------------------------------------------------------------
_RESULTS_PATH = os.path.join(_ROOT_DIR, "data", "results.json")

def _get_session_id() -> str:
    """Her ziyaretçiye tarayıcı oturumu boyunca sabit kalan benzersiz kimlik atar."""
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return session["sid"]

def _load_history(session_id: str | None = None) -> list:
    """
    data/results.json dosyasından geçmiş analiz kayıtlarını okur.
    session_id verilirse yalnızca o kullanıcıya ait kayıtlar döner.
    """
    try:
        with open(_RESULTS_PATH, encoding="utf-8") as f:
            all_records = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    if session_id:
        return [r for r in all_records if r.get("session_id") == session_id]
    return all_records

def _persist_result(record: dict) -> None:
    """
    Analiz sonucunu data/results.json'a ekler.
    Her kayıtta session_id bulunur; böylece kullanıcılar birbirinin
    geçmişini göremez.
    """
    os.makedirs(os.path.dirname(_RESULTS_PATH), exist_ok=True)
    try:
        with open(_RESULTS_PATH, encoding="utf-8") as f:
            all_records = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_records = []
    all_records.append(record)
    # Sunucu genelinde son 500 kayıt tutulur (dosya şişmesin)
    if len(all_records) > 500:
        all_records = all_records[-500:]
    with open(_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# Model meta yardımcıları (Streamlit app.py'deki ile aynı mantık)
# ---------------------------------------------------------------------------

def _load_model_meta(model_key: str) -> dict:
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

# ---------------------------------------------------------------------------
# Özet hesaplama (Streamlit app.py'deki generate_summary ile aynı)
# ---------------------------------------------------------------------------

def _generate_summary(enriched: pd.DataFrame, optimized: pd.DataFrame,
                       candidate_pool: pd.DataFrame) -> dict:
    total      = len(enriched)
    avg_churn  = float(enriched["churn_proba"].mean()) if total > 0 else 0.0
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

# ---------------------------------------------------------------------------
# Plotly figürlerini JSON'a çeviren yardımcı
# ---------------------------------------------------------------------------

def _fig_to_json(fig) -> dict:
    """Plotly Figure nesnesini frontend'in Plotly.react() için JSON'a dönüştürür."""
    return json.loads(pio.to_json(fig))

# ---------------------------------------------------------------------------
# Grafik üretim fonksiyonları (Streamlit app.py'dekilerle aynı mantık)
# ---------------------------------------------------------------------------

def _plot_risk_donut(risk_counts: pd.Series) -> dict:
    color_map = {"Yüksek": "#ef4444", "Orta": "#f59e0b", "Düşük": "#22c55e"}
    fig = go.Figure(go.Pie(
        labels=risk_counts.index, values=risk_counts.values, hole=0.60,
        marker_colors=[color_map.get(l, "#94a3b8") for l in risk_counts.index],
        textinfo="label+percent", textfont_size=12,
        hovertemplate="%{label}: %{value} müşteri (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(t=40, b=20, l=10, r=10), height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Risk Dağılımı — Aday Havuzu", font_size=13, font_color="#0f172a", x=0.5),
        annotations=[dict(
            text=f"<b>{risk_counts.sum()}</b><br><span style='font-size:11px'>aday</span>",
            x=0.5, y=0.5, font_size=15, showarrow=False, font_color="#374151"
        )],
    )
    return _fig_to_json(fig)


def _plot_action_bar(action_counts: pd.Series) -> dict:
    fig = go.Figure(go.Bar(
        x=action_counts.values, y=action_counts.index, orientation="h",
        marker=dict(color=action_counts.values,
                    colorscale=[[0, "#bfdbfe"], [1, "#1d4ed8"]], showscale=False),
        text=action_counts.values, textposition="outside",
        hovertemplate="%{y}: %{x} müşteri<extra></extra>",
    ))
    fig.update_layout(
        height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=10, l=10, r=50),
        title=dict(text="Aksiyon Dağılımı — Aday Havuzu", font_size=13, font_color="#0f172a", x=0.5),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, tickfont_size=10),
    )
    return _fig_to_json(fig)


def _plot_clv_churn_scatter(df: pd.DataFrame) -> dict:
    _med = df["estimated_clv"].median()
    fig = px.scatter(
        df, x="churn_proba", y="estimated_clv", color="risk_level",
        size="expected_loss", size_max=22,
        hover_data={"action_category": True, "MonthlyCharges": True,
                    "tenure": True, "churn_proba": ":.2f", "estimated_clv": ":,.0f"},
        color_discrete_map={"Yüksek": "#ef4444", "Orta": "#f59e0b", "Düşük": "#22c55e"},
        labels={"churn_proba": "Terk Riski", "estimated_clv": "Tahmini CLV (TL)",
                "risk_level": "Risk Seviyesi"},
        title="CLV × Churn Risk Matrisi", opacity=0.82,
    )
    fig.add_vline(x=0.5, line_dash="dot", line_color="#94a3b8", line_width=1,
                  annotation_text="Eşik: 0.50", annotation_position="top right",
                  annotation_font_size=10)
    fig.add_hline(y=_med, line_dash="dot", line_color="#94a3b8", line_width=1,
                  annotation_text=f"Medyan CLV: {_med:,.0f} TL",
                  annotation_position="right", annotation_font_size=10)
    fig.update_layout(
        height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8fafc",
        margin=dict(t=60, b=40, l=50, r=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title_text=""),
        title=dict(font_size=13, font_color="#0f172a", x=0.0),
    )
    return _fig_to_json(fig)


def _plot_roi_distribution(optimized: pd.DataFrame) -> dict:
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
                  annotation_position="right", annotation_font_size=10)
    fig.update_layout(
        height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Seçilen Müşteri ROI Dağılımı (İlk 20)",
                   font_size=13, font_color="#0f172a", x=0.0),
        xaxis=dict(title="Müşteri Sırası (ROI'ye göre)", showgrid=False),
        yaxis=dict(title="ROI (net fayda / maliyet)", showgrid=True, gridcolor="#f1f5f9"),
        margin=dict(t=60, b=50, l=60, r=30),
    )
    return _fig_to_json(fig)


def _plot_strategy_comparison(comparison_df: pd.DataFrame) -> dict:
    strategies   = comparison_df["strategy"].tolist()
    net_benefits = comparison_df["net_benefit"].tolist()
    rois         = comparison_df["avg_roi"].tolist()
    bar_colors   = ["#1d4ed8", "#64748b", "#94a3b8"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Net Fayda (TL)", x=strategies, y=net_benefits,
        marker_color=bar_colors,
        text=[f"{v:,.0f} TL" for v in net_benefits],
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
        height=390, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Strateji Karşılaştırması — Net Fayda & Ortalama ROI",
                   font_size=13, font_color="#0f172a"),
        yaxis=dict(title="Net Fayda (TL)", showgrid=True, gridcolor="#f1f5f9"),
        yaxis2=dict(title="Ort. ROI", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=70, b=50), barmode="group",
    )
    return _fig_to_json(fig)


def _plot_history_line(history: list) -> dict:
    """
    Geçmiş analiz kayıtlarından zaman içindeki net fayda eğrisini çizer.
    data/results.json'daki kayıtları kullanır.
    """
    if not history:
        return {}
    dates    = [h.get("timestamp", "")[:10] for h in history]
    net_vals = [h.get("selected_net_benefit", 0) for h in history]
    rois     = [h.get("avg_roi", 0) for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=net_vals, name="Net Fayda (TL)",
        mode="lines+markers",
        line=dict(color="#1d4ed8", width=2),
        marker=dict(size=7),
        hovertemplate="%{x}<br>Net Fayda: %{y:,.0f} TL<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=rois, name="Ort. ROI",
        mode="lines+markers",
        line=dict(color="#f59e0b", width=2, dash="dot"),
        marker=dict(size=7),
        yaxis="y2",
        hovertemplate="%{x}<br>ROI: %{y:.2f}x<extra></extra>",
    ))
    fig.update_layout(
        height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Geçmiş Analiz Performansları — Zaman İçindeki Etki",
                   font_size=13, font_color="#0f172a"),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(title="Net Fayda (TL)", showgrid=True, gridcolor="#f1f5f9"),
        yaxis2=dict(title="Ort. ROI", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40),
    )
    return _fig_to_json(fig)

# ===========================================================================
# Flask Route'ları
# ===========================================================================

@app.route("/")
def index():
    """Ana dashboard sayfasını render eder."""
    # Model meta bilgileri sidebar için önceden yüklenir
    cat_meta = _load_model_meta("catboost")
    xgb_meta = _load_model_meta("xgboost")
    return render_template("index.html",
                           cat_meta=cat_meta, xgb_meta=xgb_meta,
                           cat_threshold=_load_threshold("catboost"),
                           xgb_threshold=_load_threshold("xgboost"))


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Ana analiz endpointi.
    Multipart form: file (CSV) + form alanları (model_key, max_budget, candidate_ratio)

    Pipeline:
      Doğrulama → Tahmin → SHAP → Ajan → Optimizasyon → Strateji Değerlendirme → Persistence
    """
    # --- Dosya alımı --------------------------------------------------------
    if "file" not in request.files:
        return jsonify({"error": "CSV dosyası gereklidir."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Dosya seçilmedi."}), 400

    try:
        raw_df = pd.read_csv(io.BytesIO(file.read()))
    except Exception as err:
        return jsonify({"error": f"CSV okunamadı: {err}"}), 400

    # --- Parametreler -------------------------------------------------------
    model_key        = request.form.get("model_key", "xgboost")
    max_budget       = float(request.form.get("max_budget", 2000))
    candidate_ratio  = float(request.form.get("candidate_ratio", 0.20))
    use_llm          = request.form.get("use_llm", "0") == "1"

    if max_budget <= 0:
        return jsonify({"error": "Kampanya bütçesi sıfırdan büyük olmalıdır."}), 400

    # --- Veri doğrulama -----------------------------------------------------
    val_report = DataValidator().validate(raw_df)
    if not val_report.is_valid:
        return jsonify({"error": val_report.format_errors()}), 422

    # --- Bileşenleri al -----------------------------------------------------
    try:
        comps = _get_components()
    except Exception as err:
        return jsonify({"error": f"Model yükleme hatası: {err}"}), 500

    router    = comps["router"]
    agent     = comps["agent"]
    optimizer = comps["optimizer"]
    evaluator = comps["evaluator"]
    shap_svc  = comps["shap_svc"]
    ab_sim    = comps["ab_sim"]

    # --- Analiz pipeline ----------------------------------------------------
    try:
        # Model tarafından kullanılan bilinen sütunlar
        _MODEL_COLS = {
            "tenure", "MonthlyCharges", "TotalCharges", "Contract",
            "PaymentMethod", "InternetService", "gender", "SeniorCitizen",
            "Partner", "Dependents", "PhoneService", "MultipleLines",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling",
            "Churn",
        }
        # Modelin bilmediği tüm sütunları sakla (customerID, ad, telefon, e-posta vb.)
        _extra_col_names = [c for c in raw_df.columns if c not in _MODEL_COLS]
        _extra_cols = raw_df[_extra_col_names].reset_index(drop=True) if _extra_col_names else None

        # 1 — Tahmin
        predictions = router.predict(raw_df, model=model_key)

        # Ekstra sütunları tahmin sonucunun başına ekle
        if _extra_cols is not None:
            predictions = predictions.reset_index(drop=True)
            for col in reversed(_extra_col_names):
                predictions.insert(0, col, _extra_cols[col].values)

        # 2 — SHAP
        shap_fitted = False
        try:
            model_obj    = router.cat_service.model if model_key == "catboost" else router.xgb_service.model
            _meta_cols   = set(_extra_col_names or []) | {"churn_proba", "predicted_churn", "threshold_used", "model_name"}
            feature_cols = [c for c in predictions.columns if c not in _meta_cols]
            shap_svc.fit(model_obj, predictions[feature_cols])
            agent.set_shap_service(shap_svc, feature_cols)
            shap_fitted = True
        except Exception as shap_err:
            logger.warning("SHAP başlatılamadı: %s", shap_err)

        # 3 — Ajan
        enriched = agent.run(predictions)

        # 4 — Aday havuzu + optimizasyon
        cand_count     = min(max(10, math.ceil(len(enriched) * candidate_ratio)), len(enriched))
        candidate_pool = enriched[enriched["risk_level"].isin(["Yüksek", "Orta"])].head(cand_count).copy()
        if len(candidate_pool) == 0:
            return jsonify({"error": "Aday havuzunda yüksek veya orta riskli müşteri bulunamadı."}), 422

        optimized = optimizer.select_by_constraints(candidate_pool, max_budget=max_budget)
        summary   = _generate_summary(enriched, optimized, candidate_pool)

        # 5 — Strateji karşılaştırması
        comparison_df = evaluator.compare_all(optimized, candidate_pool, max_budget)
        advantage     = evaluator.agent_advantage_summary(comparison_df)

        # 6 — A/B testi simülasyonu (akademik katman)
        ab_results = ab_sim.run_simulation(optimized, candidate_pool) if len(optimized) >= 2 else {}

    except Exception as err:
        logger.exception("Pipeline hatası")
        return jsonify({"error": f"Analiz hatası: {err}"}), 500

    # --- Grafik üretimi -----------------------------------------------------
    charts = {
        "risk_donut":    _plot_risk_donut(candidate_pool["risk_level"].value_counts()),
        "action_bar":    _plot_action_bar(candidate_pool["action_category"].value_counts()),
        "clv_scatter":   _plot_clv_churn_scatter(candidate_pool),
        "strategy_comp": _plot_strategy_comparison(comparison_df),
    }
    if len(optimized) > 0:
        charts["roi_dist"] = _plot_roi_distribution(optimized)

    # SHAP grafiği + bireysel açıklamalar
    shap_rows = {}
    if shap_fitted:
        _non_feature = set(_extra_col_names or []) | {
            "churn_proba", "predicted_churn", "threshold_used", "model_name",
            "risk_level", "action_category", "action_detail", "action_channel",
            "action_priority", "personalization_note", "action", "action_reason",
            "retention_uplift", "expected_saved_value", "estimated_clv",
            "expected_loss", "priority_score", "ranking_score", "lifetime_value",
        }
        shap_cols = [c for c in candidate_pool.columns if c not in _non_feature]
        shap_fig  = shap_svc.plot_shap_bar(candidate_pool[shap_cols], top_n=10,
                                            title="Churn Tahminine En Çok Katkıda Bulunan 10 Değişken")
        if shap_fig:
            charts["shap_bar"] = _fig_to_json(shap_fig)

        # Bireysel SHAP — her aday için waterfall verisi
        feat_data = predictions[feature_cols]
        id_col = _extra_col_names[0] if _extra_col_names else None
        for idx in candidate_pool.index:
            if idx not in feat_data.index:
                continue
            try:
                wd = shap_svc.waterfall_data(feat_data.loc[idx], feature_cols, top_n=8)
                if wd is not None:
                    cust_id = str(candidate_pool.loc[idx, id_col]) if id_col else str(idx)
                    shap_rows[cust_id] = wd[["label", "shap_value", "direction"]].to_dict(orient="records")
            except Exception:
                pass

    # --- Tablo verileri (JSON serializasyon için NaN temizleme) -------------
    def _df_to_records(df: pd.DataFrame, cols: list) -> list:
        available = [c for c in cols if c in df.columns]
        return (
            df[available]
            .fillna("")
            .round(4)
            .to_dict(orient="records")
        )

    _id_cols = _extra_col_names or []
    CAND_COLS = [
        *_id_cols, "model_name", "tenure", "MonthlyCharges", "estimated_clv", "churn_proba",
        "expected_loss", "priority_score", "risk_level", "action_category",
        "action_detail", "action_channel", "action_reason", "personalization_note",
    ]
    OPT_COLS = [
        *_id_cols, "model_name", "MonthlyCharges", "estimated_clv", "churn_proba",
        "expected_loss", "expected_saved_value", "offer_cost", "net_benefit",
        "roi", "risk_level", "action_category", "action_detail", "action_channel", "action_reason",
        *(["llm_comment", "llm_source"] if use_llm else []),
    ]
    COMP_COLS = [
        "strategy", "selected_count", "total_cost", "expected_saved",
        "net_benefit", "avg_roi", "cost_efficiency", "precision_at_k",
    ]

    # --- LLM müşteri yorumları (opsiyonel) ----------------------------------
    llm_actually_used = False
    if use_llm:
        # Yerel ortamda llama.cpp sunucusunu otomatik başlat
        try:
            from llama_server_manager import is_server_running, start_server
            if not is_server_running():
                logger.info("llama.cpp sunucusu başlatılıyor (Qwen 2.5)...")
                ok = start_server()
                if ok:
                    logger.info("llama.cpp sunucusu hazır.")
                else:
                    logger.warning(
                        "llama.cpp sunucusu başlatılamadı — "
                        "model dosyası veya sunucu yolu bulunamıyor. "
                        "Kural tabanlı yorumlar kullanılacak."
                    )
        except Exception as srv_err:
            # Bulut ortamında (Railway vb.) llama_server_manager çalışmaz — normal
            logger.info("llama server yöneticisi devre dışı (%s). Kural tabanlı mod.", srv_err)

        try:
            llm_svc = LLMService()
            optimized, llm_actually_used = llm_svc.add_llm_comment(optimized, limit=6)
            logger.info(
                "Yorumlar eklendi — kaynak: %s",
                "Qwen 2.5 (AI)" if llm_actually_used else "kural tabanlı fallback"
            )
        except Exception as llm_err:
            logger.warning("LLM yorum ekleme hatası: %s", llm_err)

    # --- Portföy yorum metni ------------------------------------------------
    portfolio_comment  = rule_based_portfolio_summary(summary)
    strategy_comment   = rule_based_strategy_comment(comparison_df, advantage)

    # --- Kalıcılık (Persistence) — data/results.json'a yaz -----------------
    run_ts = datetime.now().isoformat()
    record = {
        "timestamp":             run_ts,
        "session_id":            _get_session_id(),
        "model_key":             model_key,
        "total_customers":       summary["total_customers"],
        "high_risk_count":       summary["high_risk_count"],
        "avg_churn":             round(summary["avg_churn"], 4),
        "selected_count":        summary["selected_count"],
        "selected_budget":       round(summary["selected_budget"], 2),
        "selected_net_benefit":  round(summary["selected_net_benefit"], 2),
        "avg_roi":               round(summary["avg_roi"], 3),
        "ab_p_value":            ab_results.get("p_value"),
        "ab_power":              ab_results.get("statistical_power"),
        "ab_significant":        ab_results.get("p_value_significant"),
    }
    try:
        _persist_result(record)
    except Exception as persist_err:
        logger.warning("Persistence hatası: %s", persist_err)

    # --- CSV çıktıları — timestamped outputs/ klasörü -----------------------
    run_slug    = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs_dir = os.path.join(_ROOT_DIR, "outputs", run_slug)
    os.makedirs(outputs_dir, exist_ok=True)

    enriched.to_csv(      os.path.join(outputs_dir, "tum_sonuclar.csv"),           index=False)
    candidate_pool.to_csv(os.path.join(outputs_dir, "aday_havuzu.csv"),            index=False)
    optimized.to_csv(     os.path.join(outputs_dir, "optimizasyon.csv"),           index=False)
    comparison_df.to_csv( os.path.join(outputs_dir, "strateji_karsilastirma.csv"), index=False)

    # --- JSON yanıt ---------------------------------------------------------
    return jsonify({
        "ok":                  True,
        "run_slug":            run_slug,
        "summary":             summary,
        "advantage":           advantage,
        "portfolio_comment":   portfolio_comment,
        "strategy_comment":    strategy_comment,
        "ab_test":             ab_results,
        "charts":              charts,
        "shap_rows":           shap_rows,
        "candidate_table":     _df_to_records(candidate_pool.head(20), CAND_COLS),
        "optimized_table":     _df_to_records(optimized.head(20), OPT_COLS),
        "comparison_table":    _df_to_records(comparison_df, COMP_COLS),
        "llm_enabled":         use_llm,
        "llm_actually_used":   llm_actually_used,
        "warnings":            [],
    })


@app.route("/api/history")
def history():
    """
    Yalnızca aktif kullanıcının geçmiş analiz kayıtlarını döner.
    Session cookie ile kullanıcılar birbirinin verilerini göremez.
    """
    sid     = _get_session_id()
    records = _load_history(session_id=sid)
    chart   = _plot_history_line(records)
    return jsonify({"records": records, "chart": chart})


@app.route("/api/history", methods=["DELETE"])
def clear_history():
    """Aktif kullanıcıya ait tüm geçmiş kayıtları siler."""
    sid = _get_session_id()
    try:
        with open(_RESULTS_PATH, encoding="utf-8") as f:
            all_records = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_records = []
    kept = [r for r in all_records if r.get("session_id") != sid]
    with open(_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)
    logger.info("Geçmiş temizlendi: session %s, silinen %d kayıt", sid, len(all_records) - len(kept))
    return jsonify({"ok": True, "deleted": len(all_records) - len(kept)})


@app.route("/api/model_info")
def model_info():
    """Her iki model için meta bilgi ve eşik değerlerini döner."""
    result = {}
    for key in ("catboost", "xgboost"):
        meta = _load_model_meta(key)
        result[key] = {
            "threshold": _load_threshold(key),
            "metrics":   meta.get("metrics", {}),
            "algorithm": meta.get("algorithm", key),
        }
    return jsonify(result)


# ---------------------------------------------------------------------------
# Uygulama başlatma
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Bileşenleri önceden yükle (ilk istek gecikmesini önle)
    try:
        _get_components()
    except Exception as startup_err:
        logger.error("Bileşen ön-yüklemesi başarısız: %s", startup_err)

    app.run(host="0.0.0.0", port=5000, debug=False)
