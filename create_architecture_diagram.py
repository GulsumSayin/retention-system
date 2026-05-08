import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.font_manager as fm

# Use Segoe UI Emoji for emoji support on Windows, with DejaVu Sans as fallback
plt.rcParams['font.family'] = ['Segoe UI Emoji', 'Segoe UI', 'DejaVu Sans']

output_path = r"C:\Users\glsms\Desktop\retention_system\outputs\sistem_mimarisi.png"

fig, ax = plt.subplots(figsize=(10, 14))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.set_xlim(0, 10)
ax.set_ylim(0, 15)
ax.set_aspect('auto')
ax.axis('off')

# Thin light gray border around entire figure
fig.patch.set_linewidth(1.5)
for spine in ax.spines.values():
    spine.set_visible(False)

rect_border = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                              transform=fig.transFigure,
                              boxstyle="square,pad=0",
                              facecolor='none',
                              edgecolor='#CCCCCC',
                              linewidth=1.5,
                              clip_on=False)
fig.add_artist(rect_border)

def draw_box(ax, cx, cy, w, h, color, label_lines, fontsize_title=11, fontsize_sub=9):
    x = cx - w / 2
    y = cy - h / 2
    patch = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.08",
                            facecolor=color,
                            edgecolor='white',
                            linewidth=1.5)
    ax.add_patch(patch)
    if len(label_lines) == 1:
        ax.text(cx, cy, label_lines[0],
                ha='center', va='center', color='white',
                fontsize=fontsize_title, fontweight='bold',
                wrap=True)
    elif len(label_lines) == 2:
        ax.text(cx, cy + 0.18, label_lines[0],
                ha='center', va='center', color='white',
                fontsize=fontsize_title, fontweight='bold')
        ax.text(cx, cy - 0.18, label_lines[1],
                ha='center', va='center', color='white',
                fontsize=fontsize_sub)
    return patch

def draw_arrow(ax, x, y_start, y_end):
    ax.annotate('', xy=(x, y_end + 0.45), xytext=(x, y_start - 0.45),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=2))

# --- Title ---
ax.text(5, 14.3,
        "Müşteri Tutundurma Zekâsı Platformu\n7 Katmanlı İşlem Mimarisi",
        ha='center', va='center', color='#2C3E50',
        fontsize=14, fontweight='bold', linespacing=1.5)

# --- Layer 0: CSV Input ---
draw_box(ax, 5, 13.2, 8, 0.9, '#4A90D9',
         ["📂 CSV Girişi", "Müşteri Verisi"])

# Arrow 0 -> 1
draw_arrow(ax, 5, 13.2, 11.7)

# --- Layer 1: Validation ---
draw_box(ax, 5, 11.7, 8, 0.9, '#5BA05A',
         ["🔍 Katman 1 — Doğrulama", "Sütun kontrolü · Veri tipi · Değer aralığı"])

# Arrow 1 -> 2
draw_arrow(ax, 5, 11.7, 10.2)

# --- Layer 2: Preprocessing ---
draw_box(ax, 5, 10.2, 8, 0.9, '#5BA05A',
         ["⚙️ Katman 2 — Ön İşleme", "Eksik değer · Özellik mühendisliği · Kodlama"])

# Arrow 2 -> 3 (split into two)
# Left arrow to CatBoost
ax.annotate('', xy=(3.1, 8.7 + 0.45), xytext=(5, 10.2 - 0.45),
            arrowprops=dict(arrowstyle='->', color='#555555', lw=2))
# Right arrow to XGBoost
ax.annotate('', xy=(6.9, 8.7 + 0.45), xytext=(5, 10.2 - 0.45),
            arrowprops=dict(arrowstyle='->', color='#555555', lw=2))

# Layer 3 label above boxes
ax.text(5, 9.3, "Katman 3 — Model Tahmini",
        ha='center', va='center', color='#2C3E50',
        fontsize=9, fontstyle='italic')

# --- Layer 3: Two model boxes ---
# Left: CatBoost  (width=3.8, gap=0.4, so left center at 5 - 0.2 - 1.9 = 2.9, right center at 5 + 0.2 + 1.9 = 7.1)
left_cx = 5 - 0.2 - 1.9   # = 2.9
right_cx = 5 + 0.2 + 1.9  # = 7.1

# CatBoost box (3 lines)
x_l = left_cx - 3.8 / 2
y_l = 8.7 - 0.9 / 2
patch_cb = FancyBboxPatch((x_l, y_l), 3.8, 0.9,
                           boxstyle="round,pad=0.08",
                           facecolor='#1f77b4',
                           edgecolor='white',
                           linewidth=1.5)
ax.add_patch(patch_cb)
ax.text(left_cx, 8.7 + 0.2, "🏆 CatBoost",
        ha='center', va='center', color='white', fontsize=10, fontweight='bold')
ax.text(left_cx, 8.7 - 0.05, "Champion Model",
        ha='center', va='center', color='white', fontsize=8.5)
ax.text(left_cx, 8.7 - 0.28, "ROC-AUC: 0.842",
        ha='center', va='center', color='white', fontsize=8.5)

# XGBoost box (3 lines)
x_r = right_cx - 3.8 / 2
y_r = 8.7 - 0.9 / 2
patch_xg = FancyBboxPatch((x_r, y_r), 3.8, 0.9,
                           boxstyle="round,pad=0.08",
                           facecolor='#ff7f0e',
                           edgecolor='white',
                           linewidth=1.5)
ax.add_patch(patch_xg)
ax.text(right_cx, 8.7 + 0.2, "🥈 XGBoost",
        ha='center', va='center', color='white', fontsize=10, fontweight='bold')
ax.text(right_cx, 8.7 - 0.05, "Challenger Model",
        ha='center', va='center', color='white', fontsize=8.5)
ax.text(right_cx, 8.7 - 0.28, "ROC-AUC: 0.840",
        ha='center', va='center', color='white', fontsize=8.5)

# Arrows from model boxes down to Layer 4
ax.annotate('', xy=(5, 7.2 + 0.45), xytext=(left_cx, 8.7 - 0.45),
            arrowprops=dict(arrowstyle='->', color='#555555', lw=2))
ax.annotate('', xy=(5, 7.2 + 0.45), xytext=(right_cx, 8.7 - 0.45),
            arrowprops=dict(arrowstyle='->', color='#555555', lw=2))

# Small annotation on right side for Layer 3
ax.annotate("Kullanıcı seçimine\ngöre aktif model",
            xy=(right_cx + 1.9, 8.7),
            xytext=(9.3, 8.7),
            ha='left', va='center', fontsize=7.5, color='#555555',
            arrowprops=dict(arrowstyle='->', color='#888888', lw=1))

# --- Layer 4: Agent Decision ---
draw_box(ax, 5, 7.2, 8, 0.9, '#9B59B6',
         ["🎯 Katman 4 — Ajan Kararı", "Risk sınıflandırma · Aksiyon atama · Kanal seçimi"])

# Arrow 4 -> 5
draw_arrow(ax, 5, 7.2, 5.7)

# --- Layer 5: Budget Optimization ---
draw_box(ax, 5, 5.7, 8, 0.9, '#E67E22',
         ["💰 Katman 5 — Bütçe Optimizasyonu", "CLV hesabı · ROI skoru · Greedy Knapsack"])

# Arrow 5 -> 6
draw_arrow(ax, 5, 5.7, 4.2)

# --- Layer 6: Statistical Evaluation ---
draw_box(ax, 5, 4.2, 8, 0.9, '#E74C3C',
         ["📊 Katman 6 — İstatistiksel Değerlendirme", "DeLong AUC · McNemar · Bootstrap · A/B Simülasyon"])

# Arrow 6 -> 7
draw_arrow(ax, 5, 4.2, 2.7)

# --- Layer 7: Web Interface / Output ---
draw_box(ax, 5, 2.7, 8, 0.9, '#2C3E50',
         ["🌐 Katman 7 — Web Arayüzü ve Sunum", "Flask REST API · Plotly görselleştirme · Render dağıtımı"])

plt.tight_layout(pad=0.5)
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Diagram saved to: {output_path}")
