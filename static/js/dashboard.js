/**
 * dashboard.js — Müşteri Tutundurma Zekâsı Platformu
 */

"use strict";

/* ==========================================================================
   DOM referansları
   ========================================================================== */
const fileInput      = document.getElementById("fileInput");
const uploadArea     = document.getElementById("uploadArea");
const fileInfo       = document.getElementById("fileInfo");
const analyzeBtn     = document.getElementById("analyzeBtn");
const btnText        = document.getElementById("btnText");
const btnSpinner     = document.getElementById("btnSpinner");
const progressWrap   = document.getElementById("progressWrap");
const progressBar    = document.getElementById("progressBar");
const progressLabel  = document.getElementById("progressLabel");
const errorPanel     = document.getElementById("errorPanel");
const resultsSection = document.getElementById("resultsSection");

/* ==========================================================================
   Başlangıç
   ========================================================================== */
window.addEventListener("DOMContentLoaded", () => {
  loadModelInfo();
  loadHistory();
  setupModelRadio();
  setupSidebarToggle();
  setupTooltips();
  setupHistoryToggle();
  setupDownloadButtons();
  setupClearHistory();
});

/* ==========================================================================
   Sidebar collapse / expand
   ========================================================================== */
function setupSidebarToggle() {
  const sidebar     = document.getElementById("sidebar");
  const collapseBtn = document.getElementById("sidebarCollapseBtn");
  const openBtn     = document.getElementById("sidebarOpenBtn");

  collapseBtn.addEventListener("click", () => {
    sidebar.classList.add("collapsed");
    openBtn.classList.remove("hidden");
  });

  openBtn.addEventListener("click", () => {
    sidebar.classList.remove("collapsed");
    openBtn.classList.add("hidden");
  });
}

/* ==========================================================================
   Tooltip sistemi
   ========================================================================== */
function setupTooltips() {
  const bubble = document.getElementById("tooltipBubble");

  document.addEventListener("mouseover", (e) => {
    const icon = e.target.closest(".info-icon[data-tip]");
    if (!icon) return;
    bubble.textContent = icon.dataset.tip;
    bubble.classList.add("visible");
    positionTooltip(e, icon);
  });

  document.addEventListener("mousemove", (e) => {
    if (!bubble.classList.contains("visible")) return;
    const icon = e.target.closest(".info-icon[data-tip]");
    if (icon) positionTooltip(e, icon);
  });

  document.addEventListener("mouseout", (e) => {
    const icon = e.target.closest(".info-icon[data-tip]");
    if (icon) bubble.classList.remove("visible");
  });
}

function positionTooltip(e, icon) {
  const bubble = document.getElementById("tooltipBubble");
  const rect   = icon.getBoundingClientRect();
  bubble.style.left = `${rect.left + window.scrollX}px`;
  bubble.style.top  = `${rect.bottom + window.scrollY + 6}px`;
}

/* ==========================================================================
   Model bilgi kartı
   ========================================================================== */
async function loadModelInfo() {
  try {
    const res  = await fetch("/api/model_info");
    const data = await res.json();
    window._modelData = data;
    updateModelCard("xgboost");
  } catch (_) { /* sunucu henüz hazır değilse sessizce geç */ }
}

function updateModelCard(key) {
  if (!window._modelData) return;
  const m = window._modelData[key] || {};
  const metrics = m.metrics || {};
  document.getElementById("mcAlgo").textContent =
    key === "catboost" ? "CatBoostClassifier" : "XGBClassifier";
  document.getElementById("mcF1").textContent =
    metrics.f1 != null ? metrics.f1.toFixed(3) : "—";
  document.getElementById("mcRoc").textContent =
    metrics.roc_auc != null ? metrics.roc_auc.toFixed(4) : "—";
  document.getElementById("mcPrauc").textContent =
    metrics.pr_auc != null ? metrics.pr_auc.toFixed(4) : "—";
}

function setupModelRadio() {
  document.querySelectorAll('input[name="modelKey"]').forEach((radio) => {
    radio.addEventListener("change", () => updateModelCard(radio.value));
  });
}

/* ==========================================================================
   Dosya yükleme: click + drag-drop
   ========================================================================== */
fileInput.addEventListener("change", onFileSelected);

uploadArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadArea.classList.add("drag-over");
});
uploadArea.addEventListener("dragleave", () => uploadArea.classList.remove("drag-over"));
uploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadArea.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) {
    fileInput.files = e.dataTransfer.files;
    showFileInfo(file);
  }
});

function onFileSelected() {
  const file = fileInput.files[0];
  if (file) showFileInfo(file);
}

function showFileInfo(file) {
  fileInfo.textContent =
    `✓ Seçilen dosya: ${file.name}  ·  ${(file.size / 1024).toFixed(0)} KB`;
  fileInfo.classList.remove("hidden");
  analyzeBtn.classList.remove("hidden");
  hideError();
}

/* ==========================================================================
   Analiz isteği
   ========================================================================== */
analyzeBtn.addEventListener("click", runAnalysis);

async function runAnalysis() {
  const file = fileInput.files[0];
  if (!file) { showError("Lütfen bir CSV dosyası seçin."); return; }

  const budget = parseFloat(document.getElementById("maxBudget").value);
  if (!budget || budget <= 0) {
    showError("Kampanya bütçesi sıfırdan büyük bir değer olmalıdır.");
    return;
  }

  setLoading(true);
  setProgress(10, "Veriler kontrol ediliyor...");
  hideError();
  resultsSection.classList.add("hidden");

  const useLlm = document.getElementById("useLlm").checked;

  const formData = new FormData();
  formData.append("file", file);
  formData.append("model_key",       getModelKey());
  formData.append("max_budget",      document.getElementById("maxBudget").value);
  formData.append("candidate_ratio", document.getElementById("candidateRatio").value);
  formData.append("use_llm",         useLlm ? "1" : "0");

  const steps = [
    [20,  "Risk skoru hesaplanıyor..."],
    [40,  "Açıklama faktörleri belirleniyor..."],
    [58,  "Müşteri öncelikleri ve aksiyonlar atanıyor..."],
    [74,  "Bütçe optimizasyonu yapılıyor..."],
    [88,  "Stratejiler karşılaştırılıyor..."],
  ];
  if (useLlm) steps.push([93, "Kişiselleştirilmiş yorumlar hazırlanıyor..."]);
  let stepIdx = 0;
  const progressTimer = setInterval(() => {
    if (stepIdx < steps.length) {
      const [pct, label] = steps[stepIdx++];
      setProgress(pct, label);
    }
  }, 900);

  try {
    const res  = await fetch("/api/analyze", { method: "POST", body: formData });
    const data = await res.json();
    clearInterval(progressTimer);

    if (!res.ok || data.error) {
      showError(data.error || "Sunucu hatası");
      setLoading(false);
      setProgress(0, "");
      return;
    }

    setProgress(100, "Analiz tamamlandı.");
    await pause(400);

    renderResults(data);
    setLoading(false);
    progressWrap.classList.add("hidden");

  } catch (err) {
    clearInterval(progressTimer);
    showError(`Bağlantı hatası: ${err.message}`);
    setLoading(false);
    setProgress(0, "");
  }
}

/* ==========================================================================
   Sonuçları render etme
   ========================================================================== */
/* ==========================================================================
   SHAP Bireysel Açıklama Modal
   ========================================================================== */
let _shapRows = {};

function openShapModal(custId, profileData) {
  const rows = _shapRows[custId];
  if (!rows || rows.length === 0) return;

  document.getElementById("shapModalTitle").textContent = `SHAP Açıklaması — ${custId}`;

  // Profil özeti
  const profile = document.getElementById("shapModalProfile");
  if (profileData) {
    const items = [
      ["Terk Riski",      profileData.churn_proba != null ? `%${(parseFloat(profileData.churn_proba)*100).toFixed(1)}` : null],
      ["Müşteri Değeri",  profileData.estimated_clv != null ? `${fmtTL(profileData.estimated_clv)} TL` : null],
      ["Risk Seviyesi",   profileData.risk_level || null],
      ["Aksiyon",         profileData.action_category || null],
    ].filter(([,v]) => v);
    profile.innerHTML = items.map(([k,v]) =>
      `<span style="background:#f1f5f9;padding:3px 10px;border-radius:20px;"><strong>${k}:</strong> ${escHtml(String(v))}</span>`
    ).join("");
  } else {
    profile.innerHTML = "";
  }

  const labels = rows.map(r => r.label);
  const values = rows.map(r => r.shap_value);
  const colors = values.map(v => v > 0 ? "#ef4444" : "#22c55e");

  Plotly.newPlot("shapModalChart", [{
    type: "bar", orientation: "h",
    x: values, y: labels,
    marker: { color: colors },
    text: values.map(v => v.toFixed(3)),
    textposition: "outside",
    hovertemplate: "%{y}: %{x:.3f}<extra></extra>",
  }], {
    height: 300,
    margin: { t: 10, b: 40, l: 10, r: 70 },
    xaxis: { title: "SHAP Değeri", zeroline: true, zerolinecolor: "#cbd5e1" },
    yaxis: { automargin: true },
    paper_bgcolor: "white", plot_bgcolor: "white",
  }, { responsive: true, displayModeBar: false });

  document.getElementById("shapModal").style.display = "flex";
}

function closeShapModal() {
  document.getElementById("shapModal").style.display = "none";
}

document.addEventListener("keydown", e => { if (e.key === "Escape") closeShapModal(); });

function renderResults(data) {
  const { summary, advantage, portfolio_comment, strategy_comment,
          ab_test, charts, candidate_table, optimized_table, comparison_table,
          shap_rows } = data;

  _shapRows = shap_rows || {};

  // Portföy yorum paneli
  setInnerHTML("portfolioComment",
    `<div class="panel-title">Portföy Değerlendirmesi</div><div>${portfolio_comment}</div>`);

  // Yönetici özeti panelleri
  renderPortfolioRiskPanel(summary);
  renderOptimizationPanel(summary);

  // KPI kartları
  renderKPIs(summary, optimized_table);

  // Grafikler
  renderChart("chartRiskDonut",    charts.risk_donut);
  renderChart("chartActionBar",    charts.action_bar);
  renderChart("chartClvScatter",   charts.clv_scatter);
  renderChart("chartStrategyComp", charts.strategy_comp);

  if (charts.roi_dist) {
    renderChart("chartRoiDist", charts.roi_dist);
    document.getElementById("roiCard").classList.remove("hidden");
  }

  // Tablo başlıkları
  document.getElementById("candidateTitle").textContent =
    `Riskli Müşteri Listesi — İlk ${candidate_table.length}`;
  document.getElementById("optimizationTitle").textContent =
    `Aksiyon Planı — ${summary.selected_count} Müşteri / ${fmtTL(summary.selected_budget)} TL Bütçe`;

  // Tablolar
  renderCandidateTable(candidate_table);
  renderOptimizedTable(optimized_table);
  renderComparisonTable(comparison_table);

  // Müşteri bazlı kart analizi — sadece LLM açıksa
  renderCustomerInsights(optimized_table, data.llm_enabled, data.llm_actually_used);

  // Strateji yorum paneli
  const panelCls = advantage.agent_is_best ? "green-top" : "gray-top";
  setInnerHTML("strategyComment",
    `<div class="panel ${panelCls}">` +
    `<div class="panel-title">Strateji Değerlendirmesi</div>` +
    `<div>${strategy_comment}</div></div>`);

  // A/B testi
  renderABTest(ab_test);

  // Grafik alt yazısı
  setInnerHTML("chartCaption",
    `Grafikler <strong>öncelikli müşteri havuzu</strong> (${summary.candidate_count} müşteri, ` +
    `yüksek + orta risk) üzerinden hesaplanmıştır. ` +
    `Portföy geneli: ${fmt(summary.total_customers)} müşteri.`);

  // İndirme butonları için veriyi sakla
  window._lastAnalysis = data;
  document.getElementById("downloadSection").classList.remove("hidden");

  // Sonuçlar göster
  resultsSection.classList.remove("hidden");
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });

  // Geçmişi güncelle
  loadHistory();
}

/* ==========================================================================
   Müşteri Bazlı Analizler — yalnızca LLM aktifken gösterilir
   ========================================================================== */
function renderCustomerInsights(rows, llmEnabled, llmActuallyUsed) {
  const panel = document.getElementById("customerInsightPanel");
  if (!panel) return;

  if (!llmEnabled || !rows || rows.length === 0) {
    panel.classList.add("hidden");
    return;
  }

  const display = rows.slice(0, 6);

  const cards = display.map(r => {
    const comment    = r.llm_comment && String(r.llm_comment).trim() ? String(r.llm_comment) : null;
    const isAI       = r.llm_source === "ai";
    const riskCls    = r.risk_level === "Yüksek" ? "risk-tag-high" : "risk-tag-mid";
    const headerTop  = [
      r.model_name ? escHtml(String(r.model_name)).toUpperCase() : "",
      r.action_channel && r.action_channel !== "Yok" ? escHtml(String(r.action_channel)).toUpperCase() : ""
    ].filter(Boolean).join(" · ");

    const commentBadge = comment
      ? (isAI
          ? '<span class="llm-badge">Qwen 2.5 Yorumu</span>'
          : '<span class="llm-badge">Analist Yorumu</span>')
      : "";

    return `
    <div class="ci-card">
      <div class="ci-header">${headerTop}</div>
      <div class="ci-title">${escHtml(String(r.action_detail || "—"))}</div>
      <div class="ci-badges">
        <span class="${riskCls}">${escHtml(String(r.risk_level || ""))} Risk</span>
        ${commentBadge}
      </div>
      ${comment ? `<div class="ci-text">${escHtml(comment)}</div>` : ""}
      <div class="ci-metrics">
        <span>Terk Riski <strong>%${(parseFloat(r.churn_proba || 0) * 100).toFixed(1)}</strong></span>
        <span>Müşteri Değeri <strong>${fmtTL(r.estimated_clv)} TL</strong></span>
        <span>Yatırım Getirisi <strong>${parseFloat(r.roi || 0).toFixed(1)}x</strong></span>
        <span>Net Fayda <strong>${fmtTL(r.net_benefit)} TL</strong></span>
      </div>
    </div>`;
  }).join("");

  const headerBadge = llmActuallyUsed
    ? '<span class="llm-count">Qwen 2.5 aktif</span>'
    : '<span class="llm-count">Analiz yorumları</span>';

  const caption = llmActuallyUsed
    ? "Her müşteri için kişiselleştirilmiş yorum ve öneri Qwen 2.5 tarafından otomatik oluşturulmuştur."
    : "Her müşteri için risk profili, sözleşme yapısı ve aksiyon gerekçesi birlikte değerlendirilerek kişiselleştirilmiş yorum hazırlanmıştır.";

  panel.innerHTML = `
    <div class="ci-wrap">
      <div class="ci-wrap-title">
        Müşteri Bazlı Analizler ${headerBadge}
      </div>
      <p class="section-caption" style="margin-bottom:14px">${caption}</p>
      <div class="ci-list">${cards}</div>
    </div>`;
  panel.classList.remove("hidden");
}

/* ==========================================================================
   Panel render yardımcıları
   ========================================================================== */
function renderPortfolioRiskPanel(s) {
  const cls = s.avg_churn >= 0.60 ? "red-top" : s.avg_churn >= 0.40 ? "blue-top" : "green-top";
  setInnerHTML("portfolioRiskPanel", `
    <div class="panel ${cls}">
      <div class="panel-title">Portföy Riski</div>
      <div>
        <strong>${s.portfolio_risk}</strong><br>
        Toplam <strong>${fmt(s.total_customers)}</strong> müşteri —
        <span style="color:#ef4444"><strong>${fmt(s.high_risk_count)} yüksek</strong></span>,
        <span style="color:#f59e0b"><strong>${fmt(s.medium_risk_count)} orta</strong></span>,
        <span style="color:#16a34a"><strong>${fmt(s.low_risk_count)} düşük</strong></span> riskli.<br>
        Ortalama terk riski: <strong>%${(s.avg_churn * 100).toFixed(1)}</strong>
        &nbsp;·&nbsp; Aciliyet: <strong>${s.urgency}</strong>
      </div>
    </div>`);
}

function renderOptimizationPanel(s) {
  setInnerHTML("optimizationPanel", `
    <div class="panel blue-top">
      <div class="panel-title">Aksiyon Planı Özeti</div>
      <div>
        <strong>${fmt(s.selected_count)}</strong> müşteri seçildi &nbsp;·&nbsp;
        Kampanya maliyeti: <strong>${fmtTL(s.selected_budget)} TL</strong><br>
        Beklenen geri kazanım: <strong>${fmtTL(s.selected_expected_saved)} TL</strong><br>
        Net fayda: <strong>${fmtTL(s.selected_net_benefit)} TL</strong>
        &nbsp;·&nbsp; Ort. yatırım getirisi: <strong>${s.avg_roi.toFixed(2)}x</strong><br>
        En sık aksiyon: <strong>${s.top_action}</strong>
      </div>
    </div>`);
}

/* ==========================================================================
   KPI kartları
   ========================================================================== */
function renderKPIs(s, optRows) {
  const totalClv   = optRows.reduce((a, r) => a + (parseFloat(r.estimated_clv) || 0), 0);
  const totalLoss  = optRows.reduce((a, r) => a + (parseFloat(r.expected_loss) || 0), 0);
  const totalSaved = optRows.reduce((a, r) => a + (parseFloat(r.expected_saved_value) || 0), 0);
  const totalCost  = optRows.reduce((a, r) => a + (parseFloat(r.offer_cost) || 0), 0);

  const kpis = [
    { label: "Toplam Müşteri",       value: fmt(s.total_customers),               hint: "Analiz edilen toplam müşteri" },
    { label: "Ort. Kayıp Riski",     value: `%${(s.avg_churn * 100).toFixed(1)}`, hint: "Portföy geneli ortalama" },
    { label: "Öncelikli Müşteriler", value: fmt(s.candidate_count),               hint: "Yüksek ve orta riskli" },
    { label: "Aksiyon Listesindeki", value: fmt(s.selected_count),                hint: "Bütçeye uygun seçilen" },
    { label: "Portföy Değeri (CLV)", value: `${fmtTL(totalClv)} TL`,              hint: "Seçilen müşterilerin toplam değeri" },
    { label: "Beklenen Kayıp",       value: `${fmtTL(totalLoss)} TL`,             hint: "Aksiyon alınmazsa oluşacak kayıp" },
    { label: "Geri Kazanılabilir",   value: `${fmtTL(totalSaved)} TL`,            hint: "Aksiyonlarla kurtarılabilecek değer" },
    { label: "Kampanya Maliyeti",    value: `${fmtTL(totalCost)} TL`,             hint: "Toplam aksiyon maliyeti" },
  ];

  const grid = document.getElementById("kpiGrid");
  grid.innerHTML = kpis.map(k => `
    <div class="kpi-card">
      <div class="kpi-label">${k.label}</div>
      <div class="kpi-value">${k.value}</div>
      <div class="kpi-hint">${k.hint}</div>
    </div>`).join("");
}

/* ==========================================================================
   Plotly grafik render
   ========================================================================== */
function renderChart(containerId, figJson) {
  if (!figJson || !figJson.data) return;
  const el = document.getElementById(containerId);
  if (!el) return;
  Plotly.react(el, figJson.data, figJson.layout, { responsive: true, displayModeBar: false });
}

/* ==========================================================================
   Tablo üretimi
   ========================================================================== */
const CAND_HEADERS = {
  model_name: "Model", tenure: "Abonelik Süresi (ay)", MonthlyCharges: "Aylık Ücret",
  estimated_clv: "Müşteri Değeri", churn_proba: "Kayıp Riski",
  expected_loss: "Beklenen Kayıp", priority_score: "Öncelik Skoru",
  risk_level: "Risk", action_category: "Aksiyon Türü",
  action_detail: "Önerilen Aksiyon", action_channel: "İletişim Kanalı",
  action_reason: "Neden Riskli?", personalization_note: "Temsilci Notu",
};

const OPT_HEADERS = {
  model_name: "Model", MonthlyCharges: "Aylık Ücret",
  estimated_clv: "Müşteri Değeri", churn_proba: "Kayıp Riski",
  expected_loss: "Beklenen Kayıp", expected_saved_value: "Geri Kazanım",
  offer_cost: "Aksiyon Maliyeti", net_benefit: "Net Fayda",
  roi: "Yatırım Getirisi", risk_level: "Risk",
  action_category: "Aksiyon Türü", action_detail: "Önerilen Aksiyon",
  action_channel: "İletişim Kanalı", action_reason: "Neden Riskli?",
};

const COMP_HEADERS = {
  strategy: "Strateji", selected_count: "Seçilen",
  total_cost: "Maliyet", expected_saved: "Geri Kazanım",
  net_benefit: "Net Fayda", avg_roi: "Ort. Getiri",
  cost_efficiency: "Maliyet Verimliliği", precision_at_k: "Doğruluk",
};

function buildExtraHeaders(rows, knownHeaders) {
  if (!rows || rows.length === 0) return {};
  const known = new Set(Object.keys(knownHeaders));
  const extra = {};
  Object.keys(rows[0]).forEach(k => {
    if (!known.has(k)) extra[k] = k; // sütun adını başlık olarak kullan
  });
  return extra;
}

function renderCandidateTable(rows) {
  const extra   = buildExtraHeaders(rows, CAND_HEADERS);
  const headers = { ...extra, ...CAND_HEADERS };
  renderTable("candidateTableWrap", rows, headers, formatCandidateCell);
}

function renderOptimizedTable(rows) {
  const extra   = buildExtraHeaders(rows, OPT_HEADERS);
  const headers = { ...extra, ...OPT_HEADERS };
  const hasComments = rows.length && rows.some(r => r.llm_comment && String(r.llm_comment).trim());
  if (hasComments) headers.llm_comment = "🤖 AI Yorumu";
  renderTable("optimizedTableWrap", rows, headers, formatOptCell);
}

function renderComparisonTable(rows) {
  renderTable("compTableWrap", rows, COMP_HEADERS, formatCompCell);
}

function renderTable(wrapperId, rows, headers, cellFn) {
  if (!rows || rows.length === 0) {
    document.getElementById(wrapperId).innerHTML =
      '<p style="padding:14px;color:#94a3b8;font-size:0.82rem">Veri yok.</p>';
    return;
  }

  const cols  = Object.keys(headers).filter(k => rows[0].hasOwnProperty(k));
  const thead = cols.map(k => `<th>${headers[k]}</th>`).join("");
  const isOptimized = wrapperId === "optimizedTableWrap";
  const tbody = rows.map(row => {
    const idVal = row["CustomerID"] || row["customerID"] || row["CUSTOMERID"] || "";
    const clickable = isOptimized && idVal && _shapRows[idVal];
    const trStyle = clickable ? ' style="cursor:pointer;" title="SHAP açıklaması için tıklayın"' : "";
    const trClick = clickable ? ` onclick="openShapModal('${escHtml(String(idVal))}', ${JSON.stringify({churn_proba:row.churn_proba, estimated_clv:row.estimated_clv, risk_level:row.risk_level, action_category:row.action_category})})"` : "";
    return `<tr${trStyle}${trClick}>${cols.map(k => `<td title="${escHtml(String(row[k] ?? ""))}">${cellFn(k, row[k], row)}</td>`).join("")}</tr>`;
  }).join("");

  document.getElementById(wrapperId).innerHTML =
    `<div class="table-wrap"><table class="data-table">
       <thead><tr>${thead}</tr></thead>
       <tbody>${tbody}</tbody>
     </table></div>`;
}

function formatCandidateCell(key, val, row) {
  if (key === "risk_level")    return riskBadge(val);
  if (key === "churn_proba")   return `%${(parseFloat(val) * 100).toFixed(1)}`;
  if (key === "estimated_clv" || key === "expected_loss") return `${fmtTL(val)} TL`;
  if (key === "priority_score") return parseFloat(val).toFixed(1);
  if (key === "MonthlyCharges") return `${parseFloat(val).toFixed(0)} TL`;
  return escHtml(String(val ?? ""));
}

function formatOptCell(key, val) {
  if (key === "risk_level")  return riskBadge(val);
  if (key === "churn_proba") return `%${(parseFloat(val) * 100).toFixed(1)}`;
  if (["estimated_clv", "expected_loss", "expected_saved_value",
       "offer_cost", "net_benefit", "MonthlyCharges"].includes(key))
    return `${fmtTL(val)} TL`;
  if (key === "roi") return `${parseFloat(val).toFixed(2)}x`;
  if (key === "llm_comment") {
    if (!val || !String(val).trim()) return '<span style="color:#475569;font-style:italic">—</span>';
    return `<span class="llm-cell-text">${escHtml(String(val))}</span>`;
  }
  return escHtml(String(val ?? ""));
}

function formatCompCell(key, val) {
  if (["total_cost", "expected_saved", "net_benefit"].includes(key))
    return `${fmtTL(val)} TL`;
  if (key === "avg_roi" || key === "cost_efficiency")
    return `${parseFloat(val).toFixed(2)}x`;
  if (key === "precision_at_k")
    return val != null && val !== "" ? `%${(parseFloat(val) * 100).toFixed(1)}` : "—";
  return escHtml(String(val ?? ""));
}

function riskBadge(val) {
  const cls = val === "Yüksek" ? "risk-high" : val === "Orta" ? "risk-mid" : "risk-low";
  return `<span class="${cls}">${escHtml(String(val ?? ""))}</span>`;
}

/* ==========================================================================
   Strateji Doğrulama (A/B Testi) render
   ========================================================================== */
function renderABTest(ab) {
  const sec = document.getElementById("abTestSection");
  if (!ab || ab.error) {
    sec.innerHTML = `<div class="ab-interpretation warning">
      ${ab?.error || "Karşılaştırma için yeterli müşteri verisi bulunamadı (en az 2 müşteri gereklidir)."}
    </div>`;
    return;
  }

  const sig   = ab.p_value_significant;
  const pCls  = sig ? "success" : "warning";
  const pwCls = ab.power_adequate ? "success" : "warning";
  const dCls  = Math.abs(ab.cohens_d) >= 0.50 ? "success" : "neutral";

  sec.innerHTML = `
    <div class="ab-grid">
      <div class="ab-stat-card">
        <div class="ab-stat-label">İstatistiksel Doğrulama</div>
        <div class="ab-stat-value ${pCls}">${sig ? "✓ Doğrulandı" : "✗ Belirsiz"}</div>
        <div class="ab-stat-sub">${sig ? "Fark istatistiksel olarak anlamlı (p = " + ab.p_value.toFixed(4) + ")" : "Fark istatistiksel açıdan anlamlı değil (p = " + ab.p_value.toFixed(4) + ")"}</div>
      </div>
      <div class="ab-stat-card">
        <div class="ab-stat-label">Net Fayda Güven Aralığı</div>
        <div class="ab-stat-value neutral">[${fmtTL(ab.ci_95_lower)}, ${fmtTL(ab.ci_95_upper)}] TL</div>
        <div class="ab-stat-sub">%95 güven düzeyinde beklenen aralık</div>
      </div>
      <div class="ab-stat-card">
        <div class="ab-stat-label">Analiz Güvenilirliği</div>
        <div class="ab-stat-value ${pwCls}">${(ab.statistical_power * 100).toFixed(1)}%</div>
        <div class="ab-stat-sub">${ab.power_adequate ? "✓ Sonuçlar güvenilir (%80 ve üzeri)" : "✗ Daha fazla veri ile sonuçlar güçlenecek"}</div>
      </div>
      <div class="ab-stat-card">
        <div class="ab-stat-label">Strateji Etki Büyüklüğü</div>
        <div class="ab-stat-value ${dCls}">${ab.effect_magnitude}</div>
        <div class="ab-stat-sub">Cohen's d = ${ab.cohens_d.toFixed(3)}</div>
      </div>
      <div class="ab-stat-card">
        <div class="ab-stat-label">AI Stratejisi — Ort. Net Fayda</div>
        <div class="ab-stat-value success">${fmtTL(ab.mean_ai_benefit)} TL</div>
        <div class="ab-stat-sub">${ab.n_ai} müşteri üzerinden hesaplandı</div>
      </div>
      <div class="ab-stat-card">
        <div class="ab-stat-label">Geleneksel Yaklaşım — Ort. Net Fayda</div>
        <div class="ab-stat-value neutral">${fmtTL(ab.mean_trad_benefit)} TL</div>
        <div class="ab-stat-sub">${ab.n_traditional} müşteri üzerinden hesaplandı</div>
      </div>
    </div>
    <div class="ab-interpretation ${sig ? "" : "warning"}">
      <strong>Sonuç:</strong> ${buildABInterpretation(ab)}
    </div>`;
}

/* ==========================================================================
   A/B Yorum üretici — kullanıcı dostu, istatistik jargonu yok
   ========================================================================== */
function buildABInterpretation(ab) {
  const diff    = (ab.mean_ai_benefit - ab.mean_trad_benefit);
  const diffStr = `${fmtTL(Math.abs(diff))} TL`;
  const direction = diff >= 0 ? "daha yüksek" : "daha düşük";

  if (ab.p_value_significant && ab.power_adequate) {
    return `AI destekli strateji, geleneksel yaklaşıma kıyasla müşteri başına ortalama
      <strong>${diffStr} ${direction}</strong> net fayda sağlamaktadır.
      Bu fark istatistiksel olarak güvenilir şekilde doğrulanmıştır —
      sonuçlar tesadüfi değildir ve uygulamaya alınabilir.`;
  }
  if (ab.p_value_significant && !ab.power_adequate) {
    return `Sonuçlar AI stratejisi lehine görünmektedir (müşteri başına ortalama
      <strong>${diffStr} ${direction}</strong> net fayda), ancak mevcut veri hacmi
      kesin bir yorum için henüz yeterli değildir. Daha fazla müşteri verisiyle
      analizin tekrarlanması önerilir.`;
  }
  return `İki strateji arasında ölçülen fark (${diffStr}) istatistiksel açıdan
    henüz anlamlı bulunamamıştır. Bu, farkın gerçek olmadığı anlamına gelmez —
    daha geniş bir müşteri kitlesiyle analiz tekrarlandığında sonuçlar netleşecektir.`;
}

/* ==========================================================================
   Geçmiş analiz toggle
   ========================================================================== */
function setupHistoryToggle() {
  const btn     = document.getElementById("historyToggleBtn");
  const content = document.getElementById("historyContent");
  const icon    = document.getElementById("historyToggleIcon");
  if (!btn || !content) return;

  btn.addEventListener("click", () => {
    const isOpen = !content.classList.contains("collapsed");
    if (isOpen) {
      content.classList.add("collapsed");
      icon.textContent = "▼";
      btn.innerHTML = `<span id="historyToggleIcon">▼</span> Göster`;
    } else {
      content.classList.remove("collapsed");
      icon.textContent = "▲";
      btn.innerHTML = `<span id="historyToggleIcon">▲</span> Gizle`;
    }
  });
}

/* ==========================================================================
   İndirme butonları
   ========================================================================== */
function setupDownloadButtons() {
  document.getElementById("dlCandidates").addEventListener("click", () => {
    if (window._lastAnalysis) downloadCSV(window._lastAnalysis.candidate_table, "riskli-musteri-listesi.csv");
  });
  document.getElementById("dlOptimized").addEventListener("click", () => {
    if (window._lastAnalysis) downloadCSV(window._lastAnalysis.optimized_table, "aksiyon-plani.csv");
  });
  document.getElementById("dlComparison").addEventListener("click", () => {
    if (window._lastAnalysis) downloadCSV(window._lastAnalysis.comparison_table, "strateji-karsilastirma.csv");
  });
}

function downloadCSV(rows, filename) {
  if (!rows || rows.length === 0) return;
  const cols    = Object.keys(rows[0]);
  const header  = cols.join(",");
  const body    = rows.map(r =>
    cols.map(c => {
      const v = String(r[c] ?? "").replace(/"/g, '""');
      return v.includes(",") || v.includes("\n") || v.includes('"') ? `"${v}"` : v;
    }).join(",")
  ).join("\n");
  const blob = new Blob(["﻿" + header + "\n" + body], { type: "text/csv;charset=utf-8;" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

/* ==========================================================================
   Geçmiş analiz temizleme
   ========================================================================== */
function setupClearHistory() {
  const btn = document.getElementById("clearHistoryBtn");
  if (!btn) return;
  btn.addEventListener("click", async () => {
    if (!confirm("Bu oturuma ait tüm geçmiş analizler silinecek. Emin misiniz?")) return;
    try {
      const res = await fetch("/api/history", { method: "DELETE" });
      const data = await res.json();
      if (data.ok) loadHistory();
    } catch (err) {
      alert("Geçmiş silinemedi: " + err.message);
    }
  });
}

/* ==========================================================================
   Geçmiş analiz paneli
   ========================================================================== */
async function loadHistory() {
  try {
    const res  = await fetch("/api/history");
    const data = await res.json();

    if (data.chart && data.chart.data) {
      renderChart("chartHistory", data.chart);
    }

    const records = data.records || [];
    if (records.length > 0) {
      const histHeaders = {
        timestamp:            "Tarih",
        model_key:            "Model",
        total_customers:      "Müşteri",
        selected_count:       "Aksiyon Listesi",
        selected_net_benefit: "Net Fayda (TL)",
        avg_roi:              "Ort. Getiri",
        ab_p_value:           "İstatistiksel Güven",
        ab_significant:       "Doğrulandı mı?",
      };
      renderTable("historyTableWrap", records.slice(-20).reverse(), histHeaders, formatHistoryCell);
    } else {
      document.getElementById("historyTableWrap").innerHTML =
        '<p style="padding:14px;color:#94a3b8;font-size:0.82rem">Henüz kaydedilmiş analiz yok.</p>';
    }
  } catch (_) { /* sunucu hazır değil */ }
}

function formatHistoryCell(key, val) {
  if (key === "timestamp")            return escHtml(String(val ?? "").replace("T", " ").substring(0, 19));
  if (key === "selected_net_benefit") return `${fmtTL(val)} TL`;
  if (key === "avg_roi")              return val != null ? `${parseFloat(val).toFixed(2)}x` : "—";
  if (key === "ab_p_value")           return val != null ? parseFloat(val).toFixed(4) : "—";
  if (key === "ab_significant")       return val === true ? "✓ Evet" : val === false ? "✗ Hayır" : "—";
  return escHtml(String(val ?? ""));
}

/* ==========================================================================
   Yardımcı fonksiyonlar
   ========================================================================== */
function getModelKey() {
  return document.querySelector('input[name="modelKey"]:checked').value;
}

function setLoading(loading) {
  analyzeBtn.disabled = loading;
  btnText.textContent = loading ? "Analiz ediliyor..." : "Analizi Başlat";
  btnSpinner.classList.toggle("hidden", !loading);
  progressWrap.classList.toggle("hidden", !loading);
}

function setProgress(pct, label) {
  progressBar.style.width   = `${pct}%`;
  progressLabel.textContent = label;
}

function showError(msg) {
  errorPanel.textContent = msg;
  errorPanel.classList.remove("hidden");
}
function hideError() {
  errorPanel.classList.add("hidden");
}

function setInnerHTML(id, html) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = html;
}

function fmtTL(v) {
  const n = parseFloat(v) || 0;
  return n.toLocaleString("tr-TR", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function fmt(v) {
  return parseInt(v || 0).toLocaleString("tr-TR");
}

function escHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function pause(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
