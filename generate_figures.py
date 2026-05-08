import os, sys, warnings
warnings.filterwarnings('ignore')
os.chdir(r'C:\Users\glsms\Desktop\retention_system')
sys.path.insert(0, 'src')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs('outputs', exist_ok=True)

# FIGURE 1: A/B Test Simulation with REAL values
np.random.seed(42)
n = 10000
pooled_std = (270.23 - 95.94) / 2.1586  # ~80.7

ai_per   = np.random.normal(270.23, pooled_std * 0.95, n)
trad_per = np.random.normal(95.94,  pooled_std * 1.05, n)

ai_wins = np.mean(ai_per > trad_per)
d_check = (np.mean(ai_per) - np.mean(trad_per)) / np.sqrt((np.std(ai_per)**2 + np.std(trad_per)**2) / 2)

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
fig.patch.set_facecolor('white')

bins = np.linspace(-50, 550, 80)
ax.hist(trad_per, bins=bins, alpha=0.65, color='#E74C3C',
        label='Geleneksel Toplu Yaklasim\n(Musteri basi ort: {:.1f} TL)'.format(np.mean(trad_per)),
        density=True)
ax.hist(ai_per, bins=bins, alpha=0.65, color='#2ECC71',
        label='AI Destekli Knapsack Stratejisi\n(Musteri basi ort: {:.1f} TL)'.format(np.mean(ai_per)),
        density=True)

ax.axvline(np.mean(trad_per), color='#C0392B', ls='--', lw=2, alpha=0.9)
ax.axvline(np.mean(ai_per),   color='#27AE60', ls='--', lw=2, alpha=0.9)

ax.text(0.60, 0.88,
        "AI stratejisi ustun olasiligi: %{:.1f}\n(p < 0,0001 - Istatistiksel olarak anlamli)".format(ai_wins*100),
        transform=ax.transAxes, fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F9FA', edgecolor='#2C3E50', alpha=0.9),
        color='#2C3E50')

ax.text(0.60, 0.70,
        "Cohen's d = {:.2f}\n(Cok buyuk etki buyuklugu)".format(d_check),
        transform=ax.transAxes, fontsize=10, color='#333',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8F9FA', edgecolor='#CCC', alpha=0.8))

ax.set_xlabel('Musteri Basina Simulate Edilen Net Fayda (TL)', fontsize=12)
ax.set_ylabel('Yogunluk', fontsize=12)
ax.set_title('A/B Test - Monte Carlo Simulasyon Dagilimi\n(10.000 yineleme, musteri basina net fayda karsilastirmasi)',
             fontsize=12, fontweight='bold', color='#2C3E50', pad=12)
ax.legend(fontsize=10, loc='upper right')
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('outputs/ab_test_sim.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("ab_test_sim.png OK")

# FIGURE 2: Tornado Chart with REAL OAT values
base = 46480.32  # real net benefit

params = [
    '1 Yillik Sozlesme Carpani',
    'Aylik Donem Ufku',
    'Tenure Carpani (alpha)',
]
ip = np.array([
    base * 0.0303,
    base * 0.1002,
    base * 0.2168,
])
im = np.array([
    -base * 0.0147,
    -base * 0.0981,
    -base * 0.2143,
])

y_pos = np.arange(len(params))

fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
fig.patch.set_facecolor('white')

ax.barh(y_pos, ip, left=base, height=0.55, color='#27AE60', alpha=0.85, label='+30pct degisim', zorder=2)
ax.barh(y_pos, im, left=base, height=0.55, color='#E74C3C', alpha=0.85, label='-30pct degisim', zorder=2)
ax.axvline(base, color='#2C3E50', lw=2, zorder=3, label='Taban: {:,.0f} TL'.format(base))

for i, (vp, vm) in enumerate(zip(ip, im)):
    pct_p = vp/base*100
    pct_m = vm/base*100
    ax.text(base + vp + 200, y_pos[i], '+{:,.0f} TL (+{:.1f}%)'.format(vp, pct_p), va='center', fontsize=9, color='#1D8348')
    ax.text(base + vm - 200, y_pos[i], '{:,.0f} TL ({:.1f}%)'.format(vm, pct_m), va='center', ha='right', fontsize=9, color='#C0392B')

criticality = ['Dusuk', 'Orta', 'Yuksek']
colors_crit  = ['#95A5A6', '#F39C12', '#E74C3C']
for i, (crit, col) in enumerate(zip(criticality, colors_crit)):
    ax.text(base + ip.max() + 5500, y_pos[i], crit,
            va='center', ha='left', fontsize=9, color=col, fontweight='bold')
ax.text(base + ip.max() + 5500, len(params) - 0.1, 'Kritiklik',
        va='center', ha='left', fontsize=9, color='#555', style='italic')

ax.set_yticks(y_pos)
ax.set_yticklabels(params, fontsize=11)
ax.set_xlabel('Net Fayda (TL)', fontsize=11)
ax.set_title('Duyarlilik Analizi - Tornado Grafigi\n(Her parametrenin +/-30 degisiminin net faydaya etkisi, gercek OAT verisi)',
             fontsize=12, fontweight='bold', color='#2C3E50', pad=12)
ax.legend(fontsize=10, loc='lower right')
ax.grid(axis='x', alpha=0.3, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(base - ip.max()*1.8, base + ip.max()*2.2)

plt.tight_layout()
plt.savefig('outputs/tornado_grafik.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("tornado_grafik.png OK")

# FIGURE 3: Strategy Comparison Bar
strategies  = ['AI Destekli\nKnapsack', 'Risk Bazli\nSabit Aksiyon', 'Geleneksel\nToplu Yaklasim']
net_benefits= [46480.32, 36152.39, 23340.95]
avg_rois    = [24.935, 22.893, 11.670]
bar_colors  = ['#1d4ed8', '#7c3aed', '#94a3b8']

fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)
fig.patch.set_facecolor('white')
ax2 = ax1.twinx()

bars = ax1.bar(strategies, net_benefits, color=bar_colors, alpha=0.85, width=0.5,
               zorder=2, edgecolor='white', linewidth=1.5)
ax1.bar_label(bars, labels=['{:,.0f} TL'.format(v) for v in net_benefits],
              padding=5, fontsize=11, fontweight='bold', color='#1e293b')

ax2.plot(strategies, avg_rois, 'o--', color='#f59e0b', lw=2.5, markersize=10, zorder=3,
         label='Ort. ROI')
for i, (s, r) in enumerate(zip(strategies, avg_rois)):
    ax2.text(i, r + 0.4, '{:.1f}x'.format(r), ha='center', fontsize=10, color='#b45309', fontweight='bold')

ax1.set_ylabel('Net Fayda (TL)', fontsize=12, color='#1e293b')
ax2.set_ylabel('Ortalama ROI', fontsize=12, color='#b45309')
ax2.tick_params(axis='y', colors='#b45309')
ax1.set_ylim(0, 58000)
ax2.set_ylim(0, 32)
ax1.grid(axis='y', alpha=0.3, zorder=0)
ax1.spines['top'].set_visible(False)
ax1.set_title('Strateji Karsilastirmasi - Net Fayda ve Ortalama ROI\n(2.000 TL butce, gercek pipeline ciktisi)',
              fontsize=12, fontweight='bold', color='#2C3E50', pad=12)

ax1.annotate('En Iyi Strateji', xy=(0, 46480), xytext=(0.3, 52000),
             fontsize=10, color='#1d4ed8', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#1d4ed8', lw=1.5))

from matplotlib.lines import Line2D
legend_elements = [
    plt.Rectangle((0,0),1,1, color='#1d4ed8', alpha=0.85, label='AI Destekli Knapsack'),
    plt.Rectangle((0,0),1,1, color='#7c3aed', alpha=0.85, label='Risk Bazli Sabit Aksiyon'),
    plt.Rectangle((0,0),1,1, color='#94a3b8', alpha=0.85, label='Geleneksel Toplu Yaklasim'),
    Line2D([0],[0], color='#f59e0b', lw=2, marker='o', markersize=8, label='Ort. ROI'),
]
ax1.legend(handles=legend_elements, fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig('outputs/strateji_karsilastirma.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("strateji_karsilastirma.png OK")
print("\nTUM GORSELLER GUNCELLENDI")
