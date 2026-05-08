import os, sys, json, warnings
warnings.filterwarnings('ignore')
os.chdir(r'C:\Users\glsms\Desktop\retention_system')
sys.path.insert(0, 'src')
# Force UTF-8 stdout to avoid cp1254 encoding errors on Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from preprocessing import prepare_cat_input, feature_engineering
from optimization import RetentionOptimizer
from evaluation import StrategyEvaluator, ABTestSimulator
from agents import RetentionAgent
from model_router import ModelRouter

# ── Load data and models ─────────────────────────────────────────────────────
df = pd.read_csv('data/telco_churn.csv')
y  = (df['Churn'] == 'Yes').astype(int)
_, df_test, _, y_test = train_test_split(df, y, test_size=0.2, stratify=y, random_state=42)
df_test = df_test.reset_index(drop=True)

# ── Run pipeline ─────────────────────────────────────────────────────────────
router    = ModelRouter()
agent     = RetentionAgent()
optimizer = RetentionOptimizer()
evaluator = StrategyEvaluator()
ab_sim    = ABTestSimulator()

# Predict with CatBoost (Champion)
predictions = router.predict(df_test, model='catboost')
enriched    = agent.run(predictions)

# Candidate pool: top 20% by risk
import math
cand_count     = math.ceil(len(enriched) * 0.20)
candidate_pool = enriched[enriched['risk_level'].isin(['Yüksek', 'Orta'])].head(cand_count).copy()

max_budget = 2000.0
optimized  = optimizer.select_by_constraints(candidate_pool, max_budget=max_budget)

print(f"\n=== PIPELINE RESULTS ===")
print(f"Total customers : {len(enriched)}")
print(f"Candidate pool  : {len(candidate_pool)}")
print(f"Selected (opt.) : {len(optimized)}")
print(f"Total cost      : {optimized['offer_cost'].sum():.2f} TL")
print(f"Net benefit     : {optimized['net_benefit'].sum():.2f} TL")
print(f"Avg ROI         : {optimized['roi'].mean():.3f}")

# ── Strategy comparison ───────────────────────────────────────────────────────
comparison_df = evaluator.compare_all(optimized, candidate_pool, max_budget)
print(f"\n=== STRATEGY COMPARISON ===")
print(comparison_df[['strategy','selected_count','total_cost','net_benefit','avg_roi']].to_string(index=False))

print(f"\n=== FULL STRATEGY COMPARISON (all cols) ===")
print(comparison_df.to_string(index=False))

# ── A/B Test simulation ───────────────────────────────────────────────────────
ab_results = ab_sim.run_simulation(optimized, candidate_pool)
print(f"\n=== A/B TEST RESULTS ===")
for k, v in ab_results.items():
    print(f"  {k}: {v}")

# ── OAT Sensitivity analysis ─────────────────────────────────────────────────
from sensitivity_analysis import run_oat_sensitivity, sensitivity_summary_table
try:
    # run_oat_sensitivity signature: (df, perturbations=None, max_budget=2000.0)
    sensitivity = run_oat_sensitivity(candidate_pool, max_budget=max_budget)
    print(f"\n=== SENSITIVITY ANALYSIS (OAT) ===")
    print(sensitivity.to_string(index=False))
    print(f"\n=== SENSITIVITY SUMMARY TABLE ===")
    print(sensitivity_summary_table(sensitivity).to_string(index=False))
except Exception as e:
    import traceback
    print(f"Sensitivity error: {e}")
    traceback.print_exc()
    sensitivity = None

# ── Agent advantage summary ───────────────────────────────────────────────────
advantage = evaluator.agent_advantage_summary(comparison_df)
print(f"\n=== AGENT ADVANTAGE SUMMARY ===")
for k, v in advantage.items():
    print(f"  {k}: {v}")

# ── Show all columns of optimized ────────────────────────────────────────────
print(f"\n=== OPTIMIZED DF COLUMNS ===")
print(list(optimized.columns))

print(f"\n=== OPTIMIZED DF SAMPLE (first 5 rows, key cols) ===")
key_cols = [c for c in ['customerID','risk_level','action','churn_proba',
                         'offer_cost','net_benefit','roi','expected_saved_value',
                         'expected_loss','priority_score'] if c in optimized.columns]
print(optimized[key_cols].head(5).to_string(index=False))

# Save all results to JSON for inspection
def safe_convert(v):
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v

results = {
    'pipeline': {
        'total_customers': len(enriched),
        'candidate_count': len(candidate_pool),
        'selected_count': len(optimized),
        'total_cost': float(optimized['offer_cost'].sum()),
        'net_benefit': float(optimized['net_benefit'].sum()),
        'avg_roi': float(optimized['roi'].mean()),
    },
    'strategy_comparison': comparison_df.to_dict(orient='records'),
    'ab_test': {k: safe_convert(v) for k, v in ab_results.items()},
    'agent_advantage': {k: safe_convert(v) for k, v in advantage.items()},
}
if sensitivity is not None:
    results['sensitivity'] = sensitivity.to_dict(orient='records')

os.makedirs('outputs', exist_ok=True)
with open('outputs/real_pipeline_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)

print("\nSaved to outputs/real_pipeline_results.json")
print("\nDONE")
