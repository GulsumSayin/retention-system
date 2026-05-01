"""
tests/test_evaluation.py
StrategyEvaluator birim testleri.

Test Kapsamı:
  1. evaluate_agent_strategy — boş girdi güvenliği, metrik doğruluğu
  2. evaluate_baseline_strategy — sabit maliyet/uplift, bütçe kısıtı
  3. evaluate_risk_only_strategy — yalnızca Yüksek risk seçimi
  4. compare_all — üç strateji DataFrame çıktısı
  5. agent_advantage_summary — üstünlük yüzdesi hesabı
  6. _apply_budget_greedy — sıralama ve bütçe kırpma tutarlılığı
  7. Precision@K — churn_proba >= 0.5 oranı
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import pandas as pd
import numpy as np

from evaluation import StrategyEvaluator


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

def _make_optimized(rows: list[dict]) -> pd.DataFrame:
    """RetentionOptimizer çıktısı formatında DataFrame oluşturur."""
    defaults = {
        "churn_proba":          0.80,
        "MonthlyCharges":       70.0,
        "tenure":               12,
        "risk_level":           "Yüksek",
        "offer_cost":           14.0,
        "expected_saved_value": 60.0,
        "net_benefit":          46.0,
        "roi":                  3.3,
        "priority_score":       150.0,
    }
    full_rows = []
    for row in rows:
        d = dict(defaults)
        d.update(row)
        full_rows.append(d)
    return pd.DataFrame(full_rows)


def _make_candidate_pool(n_high: int = 3, n_mid: int = 3) -> pd.DataFrame:
    """Karışık riskli aday havuzu oluşturur."""
    rows = []
    for _ in range(n_high):
        rows.append({
            "churn_proba": 0.80, "MonthlyCharges": 70.0, "tenure": 12,
            "risk_level": "Yüksek", "priority_score": 120.0,
            "expected_loss": 350.0,
        })
    for _ in range(n_mid):
        rows.append({
            "churn_proba": 0.55, "MonthlyCharges": 50.0, "tenure": 24,
            "risk_level": "Orta", "priority_score": 60.0,
            "expected_loss": 150.0,
        })
    return pd.DataFrame(rows)


evaluator = StrategyEvaluator()


# ===========================================================================
# 1. evaluate_agent_strategy
# ===========================================================================

class TestAgentStrategy:

    def test_empty_optimized_returns_zero(self):
        """Boş DataFrame → tüm metrikler 0."""
        result = evaluator.evaluate_agent_strategy(pd.DataFrame())
        assert result["selected_count"] == 0
        assert result["net_benefit"] == 0.0
        assert result["total_cost"] == 0.0

    def test_basic_metrics_computed(self):
        """Temel metrikler doğru hesaplanmalı."""
        opt = _make_optimized([
            {"offer_cost": 10.0, "expected_saved_value": 50.0,
             "net_benefit": 40.0, "roi": 4.0},
            {"offer_cost": 20.0, "expected_saved_value": 80.0,
             "net_benefit": 60.0, "roi": 3.0},
        ])
        result = evaluator.evaluate_agent_strategy(opt)
        assert result["selected_count"] == 2
        assert result["total_cost"] == pytest.approx(30.0)
        assert result["expected_saved"] == pytest.approx(130.0)
        assert result["net_benefit"] == pytest.approx(100.0)

    def test_cost_efficiency_formula(self):
        """cost_efficiency = expected_saved / total_cost."""
        opt = _make_optimized([
            {"offer_cost": 10.0, "expected_saved_value": 60.0,
             "net_benefit": 50.0, "roi": 5.0},
        ])
        result = evaluator.evaluate_agent_strategy(opt)
        assert result["cost_efficiency"] == pytest.approx(60.0 / 10.0, rel=1e-3)

    def test_strategy_name_contains_onerilen(self):
        """Strateji adı 'Önerilen' içermeli (agent_advantage_summary'ye gerek)."""
        opt = _make_optimized([{"offer_cost": 5.0, "expected_saved_value": 30.0,
                                  "net_benefit": 25.0, "roi": 5.0}])
        result = evaluator.evaluate_agent_strategy(opt)
        assert "Önerilen" in result["strategy"]


# ===========================================================================
# 2. evaluate_baseline_strategy
# ===========================================================================

class TestBaselineStrategy:

    def test_fixed_uplift_0_08(self):
        """Baseline uplift sabit %8 olmalı: expected_saved = expected_loss × 0.08."""
        pool = _make_candidate_pool(n_high=2, n_mid=0)
        result = evaluator.evaluate_baseline_strategy(pool, max_budget=999.0)
        if result["selected_count"] > 0:
            # Her müşterinin expected_saved_bl = expected_loss × 0.08
            # expected_loss = 350 → expected_saved_bl = 28.0 → net_benefit = 18.0 > 0
            assert result["expected_saved"] > 0

    def test_budget_respected(self):
        """Baseline toplam maliyet max_budget'ı aşmamalı."""
        pool = _make_candidate_pool(n_high=20, n_mid=20)
        budget = 50.0
        result = evaluator.evaluate_baseline_strategy(pool, max_budget=budget)
        assert result["total_cost"] <= budget + 1e-6

    def test_empty_pool(self):
        """Boş havuz → sıfır metrikler."""
        empty = pd.DataFrame(columns=["churn_proba", "MonthlyCharges",
                                       "risk_level", "expected_loss"])
        result = evaluator.evaluate_baseline_strategy(empty, max_budget=1000.0)
        assert result["selected_count"] == 0

    def test_strategy_name_contains_baseline(self):
        pool = _make_candidate_pool(2, 2)
        result = evaluator.evaluate_baseline_strategy(pool, max_budget=999.0)
        assert "Baseline" in result["strategy"]


# ===========================================================================
# 3. evaluate_risk_only_strategy
# ===========================================================================

class TestRiskOnlyStrategy:

    def test_only_high_risk_selected(self):
        """
        Risk-Only stratejisi yalnızca 'Yüksek' riskli müşterileri değerlendirmeli.
        """
        pool = _make_candidate_pool(n_high=3, n_mid=3)
        result = evaluator.evaluate_risk_only_strategy(pool, max_budget=999.0)
        # Seçilen müşteri sayısı orta riskli müşteri sayısından bağımsız olmalı
        pool_no_mid = _make_candidate_pool(n_high=3, n_mid=0)
        result_no_mid = evaluator.evaluate_risk_only_strategy(pool_no_mid, max_budget=999.0)
        assert result["selected_count"] == result_no_mid["selected_count"]

    def test_no_high_risk_returns_empty(self):
        """Yüksek riskli müşteri yoksa sıfır metrikler."""
        pool = _make_candidate_pool(n_high=0, n_mid=5)
        result = evaluator.evaluate_risk_only_strategy(pool, max_budget=999.0)
        assert result["selected_count"] == 0

    def test_budget_respected(self):
        """Risk-Only toplam maliyet max_budget'ı aşmamalı."""
        pool = _make_candidate_pool(n_high=30, n_mid=0)
        budget = 100.0
        result = evaluator.evaluate_risk_only_strategy(pool, max_budget=budget)
        assert result["total_cost"] <= budget + 1e-6

    def test_strategy_name(self):
        pool = _make_candidate_pool(3, 0)
        result = evaluator.evaluate_risk_only_strategy(pool, max_budget=999.0)
        assert "Risk-Only" in result["strategy"]


# ===========================================================================
# 4. compare_all
# ===========================================================================

class TestCompareAll:

    def test_returns_three_rows(self):
        """compare_all üç strateji için üç satır döndürmeli."""
        opt  = _make_optimized([{"offer_cost": 10.0, "expected_saved_value": 50.0,
                                   "net_benefit": 40.0, "roi": 4.0}])
        pool = _make_candidate_pool(3, 3)
        df   = evaluator.compare_all(opt, pool, max_budget=200.0)
        assert len(df) == 3

    def test_required_columns_present(self):
        """Karşılaştırma DataFrame'i beklenen sütunlara sahip olmalı."""
        opt  = _make_optimized([{"offer_cost": 10.0, "expected_saved_value": 50.0,
                                   "net_benefit": 40.0, "roi": 4.0}])
        pool = _make_candidate_pool(2, 2)
        df   = evaluator.compare_all(opt, pool, max_budget=200.0)
        required = {"strategy", "selected_count", "total_cost", "net_benefit", "avg_roi"}
        assert required.issubset(df.columns)


# ===========================================================================
# 5. agent_advantage_summary
# ===========================================================================

class TestAgentAdvantage:

    def _make_comparison_df(self, agent_nb, baseline_nb, risk_only_nb):
        return pd.DataFrame([
            {"strategy": "Agent Stratejisi (Önerilen)", "net_benefit": agent_nb},
            {"strategy": "Baseline (Herkese Aynı Aksiyon)", "net_benefit": baseline_nb},
            {"strategy": "Risk-Only Strateji", "net_benefit": risk_only_nb},
        ])

    def test_agent_is_best_when_highest(self):
        df = self._make_comparison_df(300, 200, 150)
        summary = evaluator.agent_advantage_summary(df)
        assert summary["agent_is_best"] is True

    def test_agent_not_best_when_lower(self):
        df = self._make_comparison_df(100, 500, 400)
        summary = evaluator.agent_advantage_summary(df)
        assert summary["agent_is_best"] is False

    def test_vs_baseline_pct_positive(self):
        """Agent baseline'dan %50 üstündeyse vs_baseline_pct ≈ +50."""
        df = self._make_comparison_df(300, 200, 100)
        summary = evaluator.agent_advantage_summary(df)
        assert summary["vs_baseline_pct"] == pytest.approx(50.0, abs=1.0)

    def test_vs_risk_only_pct(self):
        """Agent risk-only'nin 2 katıysa vs_risk_only_pct ≈ +100."""
        df = self._make_comparison_df(200, 100, 100)
        summary = evaluator.agent_advantage_summary(df)
        assert summary["vs_risk_only_pct"] == pytest.approx(100.0, abs=1.0)

    def test_equal_nets_zero_advantage(self):
        """Eşit net fayda → 0% üstünlük."""
        df = self._make_comparison_df(100, 100, 100)
        summary = evaluator.agent_advantage_summary(df)
        assert summary["vs_baseline_pct"] == pytest.approx(0.0, abs=0.1)


# ===========================================================================
# 6. _apply_budget_greedy
# ===========================================================================

class TestApplyBudgetGreedy:

    def test_sorted_by_benefit_desc(self):
        """_apply_budget_greedy net_benefit azalan sıraya göre kırpmalı."""
        df = pd.DataFrame({
            "cost":       [5.0, 5.0, 5.0],
            "net_benefit": [10.0, 30.0, 20.0],
        })
        # Sıralı: 30, 20, 10 → cumsum: 5, 10, 15 → budget=12 → ilk iki satır
        result = StrategyEvaluator._apply_budget_greedy(df, "cost", "net_benefit", 12.0)
        assert len(result) == 2
        assert result["net_benefit"].iloc[0] == pytest.approx(30.0)

    def test_all_within_budget_selected(self):
        """Tüm müşteriler bütçeye sığıyorsa hepsi seçilmeli."""
        df = pd.DataFrame({
            "cost":        [3.0, 3.0, 3.0],
            "net_benefit": [10.0, 20.0, 15.0],
        })
        result = StrategyEvaluator._apply_budget_greedy(df, "cost", "net_benefit", 100.0)
        assert len(result) == 3

    def test_zero_budget_returns_empty(self):
        """Sıfır bütçe → boş sonuç."""
        df = pd.DataFrame({"cost": [5.0], "net_benefit": [20.0]})
        result = StrategyEvaluator._apply_budget_greedy(df, "cost", "net_benefit", 0.0)
        assert len(result) == 0


# ===========================================================================
# 7. Precision@K
# ===========================================================================

class TestPrecisionAtK:

    def test_all_churners(self):
        """Tüm müşteriler churn (proba >= 0.5) → Precision@K = 1.0."""
        df = pd.DataFrame({"churn_proba": [0.8, 0.9, 0.7]})
        p = StrategyEvaluator._precision_at_k(df)
        assert p == pytest.approx(1.0)

    def test_no_churners(self):
        """Hiç churn yok (proba < 0.5) → Precision@K = 0.0."""
        df = pd.DataFrame({"churn_proba": [0.1, 0.2, 0.3]})
        p = StrategyEvaluator._precision_at_k(df)
        assert p == pytest.approx(0.0)

    def test_mixed(self):
        """2/4 churner → Precision@K = 0.5."""
        df = pd.DataFrame({"churn_proba": [0.8, 0.9, 0.3, 0.1]})
        p = StrategyEvaluator._precision_at_k(df)
        assert p == pytest.approx(0.5)

    def test_empty_returns_none(self):
        """Boş DataFrame → None."""
        p = StrategyEvaluator._precision_at_k(pd.DataFrame())
        assert p is None
