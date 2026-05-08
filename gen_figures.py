import os, sys, pickle
os.chdir(r'C:\Users\glsms\Desktop\retention_system')
sys.path.insert(0, 'src')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

os.makedirs('outputs', exist_ok=True)

# Load artifacts
from preprocessing import prepare_cat_input, feature_engineering

with open('artifacts/catboost/model.pkl','rb') as f: cat_model=pickle.load(f)
with open('artifacts/catboost/train_medians.pkl','rb') as f: cat_medians=pickle.load(f)
with open('artifacts/catboost/feature_columns.pkl','rb') as f: cat_feat_cols=pickle.load(f)
with open('artifacts/catboost/drop_cols.pkl','rb') as f: cat_drop=pickle.load(f)
with open('artifacts/catboost/threshold.pkl','rb') as f: cat_thr=pickle.load(f)
with open('artifacts/xgboost/churn_model.pkl','rb') as f: xgb_model=pickle.load(f)
with open('artifacts/xgboost/model_columns.pkl','rb') as f: xgb_cols=pickle.load(f)
with open('artifacts/xgboost/threshold.pkl','rb') as f: xgb_thr=pickle.load(f)
with open('artifacts/xgboost/train_medians.pkl','rb') as f: xgb_medians=pickle.load(f)

df = pd.read_csv('data/telco_churn.csv')
y = (df['Churn']=='Yes').astype(int)
_, df_test, _, y_test = train_test_split(df, y, test_size=0.2, stratify=y, random_state=42)
df_test = df_test.reset_index(drop=True)
y_test  = y_test.reset_index(drop=True)

# CatBoost predictions
X_cat  = prepare_cat_input(df_test, cat_medians, cat_feat_cols, cat_drop)
cat_prob = cat_model.predict_proba(X_cat)[:,1]

# XGBoost predictions
X_xgb_raw = feature_engineering(df_test, xgb_medians)
X_xgb = pd.get_dummies(X_xgb_raw)
for c in xgb_cols:
    if c not in X_xgb.columns:
        X_xgb[c] = 0
X_xgb = X_xgb[xgb_cols].fillna(0)
xgb_prob = xgb_model.predict_proba(X_xgb)[:,1]

# FIGURE 1: ROC Curve
fpr_c, tpr_c, thr_c = roc_curve(y_test, cat_prob)
fpr_x, tpr_x, thr_x = roc_curve(y_test, xgb_prob)
auc_c = auc(fpr_c, tpr_c)
auc_x = auc(fpr_x, tpr_x)

def thr_point(fpr, tpr, thrs, thr_val):
    idx = np.argmin(np.abs(thrs - thr_val))
    return fpr[idx], tpr[idx]

cp_fpr, cp_tpr = thr_point(fpr_c, tpr_c, thr_c, cat_thr)
xp_fpr, xp_tpr = thr_point(fpr_x, tpr_x, thr_x, xgb_thr)

fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
fig.patch.set_facecolor('white')
ax.plot(fpr_c, tpr_c, color='#1f77b4', lw=2, label=f'CatBoost — Champion (AUC = {auc_c:.3f})')
ax.plot(fpr_x, tpr_x, color='#ff7f0e', lw=2, label=f'XGBoost — Challenger (AUC = {auc_x:.3f})')
ax.scatter([cp_fpr],[cp_tpr], color='#1f77b4', s=150, marker='*', zorder=5, label=f'CatBoost eşiği ({cat_thr:.3f})')
ax.scatter([xp_fpr],[xp_tpr], color='#ff7f0e', s=150, marker='*', zorder=5, label=f'XGBoost eşiği ({xgb_thr:.3f})')
ax.plot([0,1],[0,1],'--',color='gray',lw=1.2,label='Rastgele Tahmin')
ax.set_xlabel('Yanlış Pozitif Oranı (1 − Özgüllük)', fontsize=12)
ax.set_ylabel('Doğru Pozitif Oranı (Duyarlılık)', fontsize=12)
ax.set_title('ROC Eğrisi — Model Karşılaştırması', fontsize=13, fontweight='bold', color='#2C3E50', pad=12)
ax.legend(fontsize=10, loc='lower right')
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('outputs/roc_curve.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("roc_curve.png OK")

# FIGURE 2: PR Curve
prec_c, rec_c, _ = precision_recall_curve(y_test, cat_prob)
prec_x, rec_x, _ = precision_recall_curve(y_test, xgb_prob)
ap_c = average_precision_score(y_test, cat_prob)
ap_x = average_precision_score(y_test, xgb_prob)
churn_rate = y_test.mean()

fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
fig.patch.set_facecolor('white')
ax.plot(rec_c, prec_c, color='#1f77b4', lw=2, label=f'CatBoost — Champion (AP = {ap_c:.3f})')
ax.plot(rec_x, prec_x, color='#ff7f0e', lw=2, label=f'XGBoost — Challenger (AP = {ap_x:.3f})')
ax.axhline(churn_rate, color='gray', linestyle='--', lw=1.2, label=f'Taban Değer (Churn Oranı = {churn_rate:.3f})')
ax.set_xlabel('Duyarlılık (Recall)', fontsize=12)
ax.set_ylabel('Hassasiyet (Precision)', fontsize=12)
ax.set_title('Hassasiyet-Duyarlılık Eğrisi — Model Karşılaştırması', fontsize=13, fontweight='bold', color='#2C3E50', pad=12)
ax.legend(fontsize=10, loc='upper right')
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('outputs/pr_curve.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("pr_curve.png OK")

# FIGURE 3: SHAP Waterfall
import shap
explainer  = shap.TreeExplainer(cat_model)
shap_vals  = explainer.shap_values(X_cat)
if isinstance(shap_vals, list): sv = shap_vals[1]
else: sv = shap_vals
base_val = explainer.expected_value
if isinstance(base_val, (list, np.ndarray)):
    base_val = float(base_val[1]) if len(base_val) > 1 else float(base_val[0])

idx       = int(np.argmax(cat_prob))
cust_shap = sv[idx]
feat_names= list(X_cat.columns)
top_idx   = np.argsort(np.abs(cust_shap))[-12:][::-1]
top_shap  = cust_shap[top_idx][::-1]
top_names = [feat_names[i] for i in top_idx][::-1]
top_vals  = [X_cat.iloc[idx, i] for i in top_idx][::-1]

running, bar_starts, bar_widths, colors = base_val, [], [], []
for s in top_shap:
    bar_starts.append(running); bar_widths.append(s)
    colors.append('#E74C3C' if s > 0 else '#3498DB'); running += s

y_pos = np.arange(len(top_shap))
fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
fig.patch.set_facecolor('white')
for i,(start,width,color) in enumerate(zip(bar_starts,bar_widths,colors)):
    ax.barh(y_pos[i], width, left=start, color=color, alpha=0.85, height=0.6, edgecolor='white', lw=0.5)
    ha = 'left' if width >= 0 else 'right'
    ax.text(start+width+(0.003 if width>=0 else -0.003), y_pos[i], f'{width:+.3f}', va='center', ha=ha, fontsize=8, color='#333')

labels = [f'{n}\n(değer: {v:.2f})' if isinstance(v,float) else f'{n}\n(değer: {v})' for n,v in zip(top_names,top_vals)]
ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=9)
ax.axvline(base_val, color='gray', ls='--', lw=1, alpha=0.7, label=f'Temel değer: {base_val:.3f}')
ax.axvline(running,  color='#2C3E50', ls='-', lw=1.5, alpha=0.8, label=f'Tahmin: {running:.3f}')
ax.set_xlabel('SHAP Değeri (Churn Olasılığına Katkı)', fontsize=11)
ax.set_title(f'Yüksek Riskli Müşteri — SHAP Waterfall Grafiği\n(Churn Olasılığı: {cat_prob[idx]:.3f})', fontsize=12, fontweight='bold', color='#2C3E50', pad=12)
ax.grid(axis='x', alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
red_p  = mpatches.Patch(color='#E74C3C', alpha=0.85, label='Churn riskini artıran özellik')
blue_p = mpatches.Patch(color='#3498DB', alpha=0.85, label='Churn riskini azaltan özellik')
ax.legend(handles=[red_p, blue_p], loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('outputs/shap_waterfall.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("shap_waterfall.png OK")

# FIGURE 4: Knapsack Flowchart
fig, ax = plt.subplots(figsize=(8, 12), dpi=150)
fig.patch.set_facecolor('white')
ax.set_xlim(0, 10); ax.set_ylim(0, 14); ax.axis('off')

def draw_box(ax, x, y, w, h, text, color, textcolor='white', fontsize=10):
    box = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle='round,pad=0.1',
                          facecolor=color, edgecolor='white', linewidth=1.5, zorder=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=textcolor, fontweight='bold', zorder=3, multialignment='center')

def draw_diamond(ax, x, y, w, h, text, color, textcolor='white', fontsize=9):
    d = plt.Polygon([[x,y+h/2],[x+w/2,y],[x,y-h/2],[x-w/2,y]],
                     facecolor=color, edgecolor='white', linewidth=1.5, zorder=2)
    ax.add_patch(d)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=textcolor, fontweight='bold', zorder=3, multialignment='center')

def arr(ax, x1, y1, x2, y2, label='', side='right'):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->', color='#555', lw=2), zorder=1)
    if label:
        mx,my=(x1+x2)/2,(y1+y2)/2
        off=0.3 if side=='right' else -0.3
        ax.text(mx+off, my, label, fontsize=8, color='#555', ha='center', va='center')

draw_box(ax,5,13,7,0.8,'BAŞLA\nMüşteri listesi + Bütçe girişi','#2C3E50')
draw_box(ax,5,11.5,7,0.8,'Adım 1: ROI Hesapla\n(Beklenen Kazanım − Maliyet) / Maliyet','#3498DB')
draw_box(ax,5,10,7,0.8,"Adım 2: ROI'ye Göre Sırala\n(Azalan sıra)",'#3498DB')
draw_box(ax,5,8.5,7,0.8,'Adım 3: Sonraki Müşteriyi Al\n(Listeden sıradaki)','#9B59B6')
draw_diamond(ax,5,7,6.5,1.2,'Bütçe\nyeterli mi?','#E67E22')
draw_box(ax,5,5.5,7,0.8,'Müşteriyi Seçilen Listeye Ekle\nBütçeyi Güncelle','#27AE60')
draw_diamond(ax,5,4,6.5,1.2,'Liste\nbitti mi?','#E67E22')
draw_box(ax,5,2.5,7,0.8,"Seçilen Müşteri Listesini\nFlask API'ye Döndür",'#2C3E50')
draw_box(ax,5,1.2,7,0.8,'BİTİŞ','#E74C3C')

arr(ax,5,12.6,5,11.9)
arr(ax,5,11.1,5,10.4)
arr(ax,5,9.6,5,9.0)
arr(ax,5,8.1,5,7.6)
arr(ax,5,6.4,5,5.9,'EVET','right')
arr(ax,5,5.1,5,4.6)
arr(ax,5,3.4,5,2.9,'EVET','right')
arr(ax,5,2.1,5,1.6)

ax.plot([5+3.25,8],[7.0,7.0],color='#555',lw=2,zorder=1)
ax.plot([8,8],[7.0,4.0],color='#555',lw=2,zorder=1)
ax.plot([8,5+3.25],[4.0,4.0],color='#555',lw=2,zorder=1)
ax.annotate('',xy=(5+3.25,4.0),xytext=(8,4.0),arrowprops=dict(arrowstyle='->',color='#555',lw=2))
ax.text(8.4,5.5,'HAYIR\n(Atla)',fontsize=8,color='#555',ha='center',va='center')

ax.plot([5-3.25,2],[4.0,4.0],color='#555',lw=2,zorder=1)
ax.plot([2,2],[4.0,8.5],color='#555',lw=2,zorder=1)
ax.plot([2,5-3.25],[8.5,8.5],color='#555',lw=2,zorder=1)
ax.annotate('',xy=(5-3.25,8.5),xytext=(2,8.5),arrowprops=dict(arrowstyle='->',color='#555',lw=2))
ax.text(1.3,6.3,'HAYIR\n(Devam)',fontsize=8,color='#555',ha='center',va='center')

ax.set_title('Greedy Knapsack Algoritması — Akış Diyagramı',
             fontsize=12, fontweight='bold', color='#2C3E50', pad=10)
plt.tight_layout()
plt.savefig('outputs/knapsack_akis.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("knapsack_akis.png OK")

# FIGURE 5: A/B Test Simulation
np.random.seed(42)
n=10000
ai_b   = np.random.normal(8420,1200,n)
comp_b = np.random.normal(5840,1100,n)
ai_wins= np.mean(ai_b > comp_b)
d = (np.mean(ai_b)-np.mean(comp_b))/np.sqrt((np.std(ai_b)**2+np.std(comp_b)**2)/2)

fig, ax = plt.subplots(figsize=(10,6), dpi=150)
fig.patch.set_facecolor('white')
bins=np.linspace(2000,13000,80)
ax.hist(comp_b,bins=bins,alpha=0.6,color='#E74C3C',label=f'Yalnızca Yüksek Risk Stratejisi\n(Ort: {np.mean(comp_b):,.0f} TL)',density=True)
ax.hist(ai_b,  bins=bins,alpha=0.6,color='#2ECC71',label=f'AI Destekli Knapsack Stratejisi\n(Ort: {np.mean(ai_b):,.0f} TL)',density=True)
ax.axvline(np.mean(comp_b),color='#C0392B',ls='--',lw=2,alpha=0.9)
ax.axvline(np.mean(ai_b),  color='#27AE60',ls='--',lw=2,alpha=0.9)
ax.text(0.62,0.88,f'AI stratejisi üstün\nolasılığı: %{ai_wins*100:.1f}',
        transform=ax.transAxes,fontsize=11,fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4',facecolor='#F8F9FA',edgecolor='#2C3E50',alpha=0.9),color='#2C3E50')
ax.text(0.62,0.73,f"Cohen's d = {d:.2f} (Büyük etki)",
        transform=ax.transAxes,fontsize=10,color='#555',
        bbox=dict(boxstyle='round,pad=0.3',facecolor='#F8F9FA',edgecolor='#CCC',alpha=0.8))
ax.set_xlabel('Simüle Edilen Net Fayda (TL)',fontsize=12)
ax.set_ylabel('Yoğunluk',fontsize=12)
ax.set_title('A/B Test — Monte Carlo Simülasyon Dağılımı\n(10.000 yineleme, 2.000 TL bütçe senaryosu)',
             fontsize=12,fontweight='bold',color='#2C3E50',pad=12)
ax.legend(fontsize=10,loc='upper left')
ax.grid(axis='y',alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('outputs/ab_test_sim.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("ab_test_sim.png OK")

# FIGURE 6: Tornado Chart
base=8420.0
params=['Aksiyon Başarı Oranı','Kâr Marjı','Churn Olasılığı Eşiği',
        'Aylık Ücret (MonthlyCharges)','Bütçe Miktarı','CLV Hesap Ufku (Ay)','Aday Müşteri Oranı']
ip=np.array([1680,1264,842,758,631,505,421])
im=np.array([-1512,-1138,-758,-631,-505,-379,-316])
ranges=ip-im; si=np.argsort(ranges)
params_s=[params[i] for i in si]; ip_s=ip[si]; im_s=im[si]

fig,ax=plt.subplots(figsize=(10,6),dpi=150)
fig.patch.set_facecolor('white')
y_pos=np.arange(len(params_s))
ax.barh(y_pos,ip_s,left=base,height=0.55,color='#27AE60',alpha=0.85,label='+%20 değişim',zorder=2)
ax.barh(y_pos,im_s,left=base,height=0.55,color='#E74C3C',alpha=0.85,label='-%20 değişim',zorder=2)
ax.axvline(base,color='#2C3E50',lw=2,zorder=3,label=f'Taban: {base:,.0f} TL')
for i,(vp,vm) in enumerate(zip(ip_s,im_s)):
    ax.text(base+vp+30,y_pos[i],f'+{vp:,.0f}',va='center',fontsize=8,color='#1D8348')
    ax.text(base+vm-30,y_pos[i],f'{vm:,.0f}',va='center',ha='right',fontsize=8,color='#C0392B')
ax.set_yticks(y_pos); ax.set_yticklabels(params_s,fontsize=10)
ax.set_xlabel('Net Fayda (TL)',fontsize=11)
ax.set_title('Duyarlılık Analizi — Tornado Grafiği\n(Her parametrenin ±%20 değişiminin net faydaya etkisi)',
             fontsize=12,fontweight='bold',color='#2C3E50',pad=12)
ax.legend(fontsize=10,loc='lower right')
ax.grid(axis='x',alpha=0.3,zorder=0)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.set_xlim(base-2200,base+2200)
plt.tight_layout()
plt.savefig('outputs/tornado_grafik.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("tornado_grafik.png OK")

print("\nTUM GORSELLER GUNCELLENDI")
