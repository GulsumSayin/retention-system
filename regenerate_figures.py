"""
Regenerate A/B test simulation chart and tornado chart with real pipeline data.
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
os.chdir(r'C:\Users\glsms\Desktop\retention_system')
sys.path.insert(0, 'src')
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as scipy_stats

# ── Load real pipeline results ─────────────────────────────────────────────
with open('outputs/real_pipeline_results.json', encoding='utf-8') as f:
    results = json.load(f)

ab = results['ab_test']
strategy = results['strategy_comparison']
sensitivity_data = results.get('sensitivity', [])

OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================================================================
# 1. A/B Test Simulation Figure
# ===========================================================================
print("Generating A/B test simulation figure...")

# Reconstruct distributions from the real numbers
rng = np.random.default_rng(42)

n_ai   = ab['n_ai']         # 172
n_trad = ab['n_traditional'] # 282
mean_ai   = ab['mean_ai_benefit']    # 270.23
mean_trad = ab['mean_trad_benefit']  # 95.94
cohens_d  = ab['cohens_d']           # 2.1586

# Reconstruct std from Cohen's d and mean difference
# pooled_std = diff / cohens_d
pooled_std = (mean_ai - mean_trad) / cohens_d  # ~80.75

std_ai   = pooled_std
std_trad = pooled_std

ai_samples   = rng.normal(mean_ai,   std_ai,   n_ai)
trad_samples = rng.normal(mean_trad, std_trad, n_trad)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('A/B Test Simülasyonu: AI Destekli vs Geleneksel Strateji', fontsize=14, fontweight='bold', y=1.01)

# --- Left: Distribution comparison ---
ax = axes[0]
bins = np.linspace(min(trad_samples.min(), ai_samples.min()) - 20,
                   max(trad_samples.max(), ai_samples.max()) + 20, 40)
ax.hist(trad_samples, bins=bins, alpha=0.6, color='#ef4444', label=f'Geleneksel (n={n_trad}, ort={mean_trad:.1f} TL)', density=True)
ax.hist(ai_samples, bins=bins, alpha=0.6, color='#3b82f6', label=f'AI Destekli (n={n_ai}, ort={mean_ai:.1f} TL)', density=True)
ax.axvline(mean_trad, color='#dc2626', linestyle='--', linewidth=2)
ax.axvline(mean_ai,   color='#1d4ed8', linestyle='--', linewidth=2)
ax.set_xlabel('Müşteri Başına Net Fayda (TL)', fontsize=11)
ax.set_ylabel('Yoğunluk', fontsize=11)
ax.set_title('Net Fayda Dağılımı Karşılaştırması', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Annotation: mean difference
ax.annotate(
    f'Δ = {ab["mean_difference"]:.1f} TL\np < 0.0001\nd = {cohens_d:.2f}',
    xy=(mean_ai, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 0.008),
    fontsize=9,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#dbeafe', alpha=0.8),
)

# --- Right: Strategy comparison bar chart ---
ax2 = axes[1]
strategies = [s['strategy'] for s in strategy]
# Shorten names for display
short_names = ['AI Destekli\nAkıllı Yaklaşım', 'Geleneksel\nToplu Yaklaşım', 'Risk Bazlı\nSabit Aksiyon']
net_benefits = [s['net_benefit'] for s in strategy]
colors = ['#3b82f6', '#ef4444', '#f59e0b']
bars = ax2.bar(short_names, net_benefits, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)

for bar, val in zip(bars, net_benefits):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
             f'{val:,.0f} TL', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_ylabel('Toplam Net Fayda (TL)', fontsize=11)
ax2.set_title('Strateji Karşılaştırması — Toplam Net Fayda', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, max(net_benefits) * 1.15)

# Add ROI annotations
rois = [s['avg_roi'] for s in strategy]
for i, (bar, roi) in enumerate(zip(bars, rois)):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5,
             f'ROI: {roi:.1f}x', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

fig.tight_layout()
ab_path = os.path.join(OUTPUT_DIR, 'ab_test_sim.png')
fig.savefig(ab_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {ab_path}")

# ===========================================================================
# 2. Tornado Chart (OAT Sensitivity)
# ===========================================================================
print("Generating tornado chart...")

if sensitivity_data:
    sens_df = pd.DataFrame(sensitivity_data)

    # Filter to ±30% only
    df30 = sens_df[sens_df['perturbation_pct'].abs() == 30.0].copy()

    labels = df30['label'].unique().tolist()

    # Build pivot: for each label, get -30 and +30 change
    pivot = {}
    for label in labels:
        sub = df30[df30['label'] == label]
        neg = sub[sub['perturbation_pct'] == -30.0]['net_benefit_change_pct'].values
        pos = sub[sub['perturbation_pct'] == 30.0]['net_benefit_change_pct'].values
        pivot[label] = {
            'neg': float(neg[0]) if len(neg) else 0.0,
            'pos': float(pos[0]) if len(pos) else 0.0,
        }

    # Sort by total range (descending impact)
    sorted_labels = sorted(labels, key=lambda l: abs(pivot[l]['pos']) - abs(pivot[l]['neg']), reverse=False)

    fig, ax = plt.subplots(figsize=(10, 5))

    y_positions = range(len(sorted_labels))
    bar_height = 0.5

    for i, label in enumerate(sorted_labels):
        neg_val = pivot[label]['neg']
        pos_val = pivot[label]['pos']
        # Draw negative bar (left of 0)
        ax.barh(i, neg_val, height=bar_height, color='#ef4444', alpha=0.85, label='-30% Değişim' if i == 0 else '')
        # Draw positive bar (right of 0)
        ax.barh(i, pos_val, height=bar_height, color='#22c55e', alpha=0.85, label='+30% Değişim' if i == 0 else '')
        # Labels
        if neg_val != 0:
            ax.text(neg_val - 0.3, i, f'{neg_val:.1f}%', ha='right', va='center', fontsize=9, color='#991b1b')
        if pos_val != 0:
            ax.text(pos_val + 0.3, i, f'+{pos_val:.1f}%', ha='left', va='center', fontsize=9, color='#166534')

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(sorted_labels, fontsize=10)
    ax.axvline(0, color='black', linewidth=1.2)
    ax.set_xlabel('Net Fayda Değişimi (%)', fontsize=11)
    ax.set_title('CLV Katsayıları Duyarlılık Analizi (Tornadogram)\n±30% Pertürbasyon', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')

    red_patch  = mpatches.Patch(color='#ef4444', alpha=0.85, label='-30% Değişim')
    green_patch = mpatches.Patch(color='#22c55e', alpha=0.85, label='+30% Değişim')
    ax.legend(handles=[red_patch, green_patch], loc='lower right', fontsize=10)

    fig.tight_layout()
    tornado_path = os.path.join(OUTPUT_DIR, 'tornado_grafik.png')
    fig.savefig(tornado_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {tornado_path}")
else:
    print("  No sensitivity data — skipping tornado chart.")

# ===========================================================================
# 3. Knapsack flow diagram (strategy selected_count vs cost vs benefit)
# ===========================================================================
print("Generating knapsack optimization figure...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Knapsack Optimizasyonu: Strateji Karşılaştırması', fontsize=13, fontweight='bold')

short_names3 = ['AI Destekli', 'Geleneksel', 'Risk Bazlı']
colors3 = ['#3b82f6', '#ef4444', '#f59e0b']

# Panel 1: Selected count
ax = axes[0]
selected = [s['selected_count'] for s in strategy]
bars = ax.bar(short_names3, selected, color=colors3, alpha=0.85)
for bar, val in zip(bars, selected):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(val),
            ha='center', va='bottom', fontweight='bold')
ax.set_ylabel('Seçilen Müşteri Sayısı')
ax.set_title('Seçilen Müşteri Sayısı')
ax.grid(True, alpha=0.3, axis='y')

# Panel 2: Total cost
ax = axes[1]
costs = [s['total_cost'] for s in strategy]
bars = ax.bar(short_names3, costs, color=colors3, alpha=0.85)
ax.axhline(2000, color='black', linestyle='--', linewidth=1.5, label='Bütçe (2000 TL)')
for bar, val in zip(bars, costs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, f'{val:,.0f}',
            ha='center', va='bottom', fontweight='bold', fontsize=9)
ax.set_ylabel('Toplam Maliyet (TL)')
ax.set_title('Bütçe Kullanımı')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Panel 3: Net benefit + avg ROI dual axis
ax = axes[2]
net_b = [s['net_benefit'] for s in strategy]
rois2 = [s['avg_roi'] for s in strategy]
x = np.arange(len(short_names3))
width = 0.4
bars1 = ax.bar(x - width/2, net_b, width, color=colors3, alpha=0.85, label='Net Fayda (TL)')
ax2b = ax.twinx()
ax2b.plot(x, rois2, 'ko-', markersize=8, linewidth=2, label='Ort. ROI')
for xi, roi in zip(x, rois2):
    ax2b.text(xi, roi + 0.3, f'{roi:.1f}x', ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(short_names3)
ax.set_ylabel('Net Fayda (TL)')
ax2b.set_ylabel('Ortalama ROI')
ax.set_title('Net Fayda & ROI')
ax.grid(True, alpha=0.3, axis='y')

fig.tight_layout()
knapsack_path = os.path.join(OUTPUT_DIR, 'knapsack_akis.png')
fig.savefig(knapsack_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {knapsack_path}")

print("\nAll figures regenerated with real data.")
print(f"\nSummary of key numbers:")
print(f"  AI Strategy:         {results['strategy_comparison'][0]['net_benefit']:,.2f} TL net benefit, ROI={results['strategy_comparison'][0]['avg_roi']:.3f}, n={results['strategy_comparison'][0]['selected_count']}")
print(f"  Baseline Strategy:   {results['strategy_comparison'][1]['net_benefit']:,.2f} TL net benefit, ROI={results['strategy_comparison'][1]['avg_roi']:.3f}, n={results['strategy_comparison'][1]['selected_count']}")
print(f"  Risk-Only Strategy:  {results['strategy_comparison'][2]['net_benefit']:,.2f} TL net benefit, ROI={results['strategy_comparison'][2]['avg_roi']:.3f}, n={results['strategy_comparison'][2]['selected_count']}")
print(f"  A/B t-statistic:     {ab['t_statistic']}, p={ab['p_value']}, Cohen d={ab['cohens_d']}")
print(f"  Agent vs Baseline:   +{results['agent_advantage']['vs_baseline_pct']}%")
print(f"  Agent vs Risk-Only:  +{results['agent_advantage']['vs_risk_only_pct']}%")
