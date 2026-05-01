"""
tests/test_optimization.py
RetentionOptimizer birim testleri.

Test Kapsamı:
  1. ROI sıralaması — en yüksek ROI ilk seçilmeli
  2. Bütçe kısıtı — toplam maliyet max_budget'ı aşmamalı
  3. Boş havuz — güvenli boş DataFrame dönmeli
  4. Sıfır bütçe — hiç müşteri seçilmemeli
  5. net_benefit <= 0 filtreleme
  6. "müdahale gerekmiyor" aksiyonu hariç tutma
  7. max_customers kısıtı
  8. offer_cost_rate hesabı — ACTION_REGISTRY oran tipi
  9. Bilinmeyen aksiyon — 0.0 maliyet ve hariç tutma
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import pandas as pd
import numpy as np

from optimization import RetentionOptimizer


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

def _make_pool(rows: list[dict]) -> pd.DataFrame:
    """
    RetentionOptimizer'a girdi olacak DataFrame oluşturur.
    Zorunlu sütunlar: expected_saved_value, action, MonthlyCharges,
                      churn_proba, priority_score
    """
    defaults = {
        "churn_proba":          0.7,
        "MonthlyCharges":       70.0,
        "tenure":               12,
        "Contract":             "Month-to-month",
        "risk_level":           "Yüksek",
        "priority_score":       100.0,
        "expected_saved_value": 50.0,
        "action":               "müşteri temsilcisi ile öncelikli temas",
        "retention_uplift":     0.15,
        "IsMonthToMonth":       1,
        "ServiceIntensity":     0.5,
    }
    full_rows = []
    for row in rows:
        d = dict(defaults)
        d.update(row)
        full_rows.append(d)
    return pd.DataFrame(full_rows)


optimizer = RetentionOptimizer()


# ===========================================================================
# 1. ROI Sıralaması
# ===========================================================================

class TestROISorting:

    def test_higher_roi_selected_first(self):
        """
        İki müşteri: yüksek ROI'li ilk seçilmeli.
        Bütçe sadece bir tanesine yeterli olacak şekilde kısıtlanır.
        """
        # action: "müşteri temsilcisi ile öncelikli temas" — fixed=5.0 TL
        # expected_saved_value yüksek = yüksek ROI
        pool = _make_pool([
            {"MonthlyCharges": 50.0, "expected_saved_value": 200.0,
             "action": "müşteri temsilcisi ile öncelikli temas", "priority_score": 500},
            {"MonthlyCharges": 50.0, "expected_saved_value":  20.0,
             "action": "müşteri temsilcisi ile öncelikli temas", "priority_score": 100},
        ])
        # Maliyet = 25 TL (fixed), bütçe = 30 TL → sadece 1 müşteri sığar
        result = optimizer.select_by_constraints(pool, max_budget=30.0)
        assert len(result) == 1
        # Yüksek expected_saved → yüksek net_benefit → yüksek ROI → o seçilmeli
        assert result["expected_saved_value"].iloc[0] == pytest.approx(200.0)

    def test_roi_computed_correctly(self):
        """roi = net_benefit / offer_cost."""
        pool = _make_pool([
            {"MonthlyCharges": 100.0, "expected_saved_value": 80.0,
             "action": "%10 indirim + 12 ay taahhüt"},
        ])
        result_with_cost = optimizer.assign_offer_cost(pool)
        assert "roi" in result_with_cost.columns
        row = result_with_cost.iloc[0]
        if row["offer_cost"] > 0:
            expected_roi = row["net_benefit"] / row["offer_cost"]
            assert row["roi"] == pytest.approx(expected_roi)


# ===========================================================================
# 2. Bütçe Kısıtı
# ===========================================================================

class TestBudgetConstraint:

    def test_total_cost_within_budget(self):
        """Seçilen müşterilerin toplam maliyeti max_budget'ı aşmamalı."""
        pool = _make_pool([
            {"MonthlyCharges": 60.0, "expected_saved_value": 50.0,
             "action": "müşteri temsilcisi ile öncelikli temas"}
            for _ in range(20)
        ])
        budget = 50.0
        result = optimizer.select_by_constraints(pool, max_budget=budget)
        if len(result) > 0:
            assert result["offer_cost"].sum() <= budget + 1e-9

    def test_zero_budget_returns_empty(self):
        """max_budget = 0 → seçilecek müşteri yok."""
        pool = _make_pool([
            {"MonthlyCharges": 50.0, "expected_saved_value": 40.0,
             "action": "müşteri temsilcisi ile öncelikli temas"}
        ])
        result = optimizer.select_by_constraints(pool, max_budget=0.0)
        assert len(result) == 0

    def test_large_budget_selects_all_eligible(self):
        """Büyük bütçe, tüm uygun müşterileri seçebilmeli."""
        pool = _make_pool([
            {"MonthlyCharges": 50.0, "expected_saved_value": 100.0,
             "action": "müşteri temsilcisi ile öncelikli temas"}
            for _ in range(5)
        ])
        result = optimizer.select_by_constraints(pool, max_budget=999_999.0)
        assert len(result) == 5


# ===========================================================================
# 3. Boş ve Kenar Durumlar
# ===========================================================================

class TestEdgeCases:

    def test_empty_pool_returns_empty_df(self):
        """Boş havuz → boş DataFrame (crash olmamalı)."""
        empty = pd.DataFrame(columns=[
            "churn_proba", "MonthlyCharges", "tenure", "Contract",
            "expected_saved_value", "action", "priority_score",
        ])
        result = optimizer.select_by_constraints(empty, max_budget=1000.0)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_no_action_intervention_excluded(self):
        """'müdahale gerekmiyor' aksiyonlu müşteriler seçilmemeli."""
        pool = _make_pool([
            {"action": "müdahale gerekmiyor", "expected_saved_value": 500.0,
             "MonthlyCharges": 50.0},
            {"action": "müşteri temsilcisi ile öncelikli temas",
             "expected_saved_value": 30.0, "MonthlyCharges": 50.0},
        ])
        result = optimizer.select_by_constraints(pool, max_budget=999.0)
        actions = result["action"].tolist()
        assert "müdahale gerekmiyor" not in actions

    def test_negative_net_benefit_excluded(self):
        """net_benefit <= 0 olan müşteriler seçilmemeli."""
        # expected_saved_value = 1.0, maliyet > 1.0 → net_benefit < 0
        pool = _make_pool([
            {"expected_saved_value": 1.0, "MonthlyCharges": 200.0,
             "action": "%10 indirim + 12 ay taahhüt"},
        ])
        result = optimizer.select_by_constraints(pool, max_budget=999.0)
        assert len(result) == 0

    def test_single_customer_selected(self):
        """Tek uygun müşteri → doğru seçilmeli."""
        pool = _make_pool([
            {"expected_saved_value": 80.0, "MonthlyCharges": 60.0,
             "action": "müşteri temsilcisi ile öncelikli temas", "churn_proba": 0.9},
        ])
        result = optimizer.select_by_constraints(pool, max_budget=100.0)
        assert len(result) == 1


# ===========================================================================
# 4. max_customers Kısıtı
# ===========================================================================

class TestMaxCustomers:

    def test_max_customers_limits_selection(self):
        """max_customers=2 iken 2'den fazla seçilmemeli."""
        pool = _make_pool([
            {"expected_saved_value": 50.0, "MonthlyCharges": 50.0,
             "action": "müşteri temsilcisi ile öncelikli temas"}
            for _ in range(10)
        ])
        result = optimizer.select_by_constraints(pool, max_budget=999.0, max_customers=2)
        assert len(result) <= 2

    def test_max_customers_none_no_limit(self):
        """max_customers=None → müşteri sayısı kısıtlanmamalı."""
        pool = _make_pool([
            {"expected_saved_value": 50.0, "MonthlyCharges": 50.0,
             "action": "müşteri temsilcisi ile öncelikli temas"}
            for _ in range(5)
        ])
        result = optimizer.select_by_constraints(pool, max_budget=999.0, max_customers=None)
        assert len(result) == 5


# ===========================================================================
# 5. Maliyet Hesabı
# ===========================================================================

class TestOfferCost:

    def test_rate_cost_proportional_to_monthly_charges(self):
        """
        "rate" tipindeki aksiyon maliyeti MonthlyCharges × cost_value olmalı.
        "%10 indirim + 12 ay taahhüt" → cost_type="rate", cost_value=0.20
        """
        mc = 100.0
        pool = _make_pool([
            {"MonthlyCharges": mc, "action": "%10 indirim + 12 ay taahhüt",
             "expected_saved_value": 500.0},
        ])
        result = optimizer.assign_offer_cost(pool)
        # cost_value = 0.20 → maliyet = 100 × 0.20 = 20
        assert result["offer_cost"].iloc[0] == pytest.approx(mc * 0.20)

    def test_fixed_cost_independent_of_monthly(self):
        """
        "fixed" tipindeki aksiyon maliyeti MonthlyCharges'a bağımsız olmalı.
        "ilk dönem memnuniyet araması" → cost_type="fixed", cost_value=5.0
        """
        pool_low  = _make_pool([{"MonthlyCharges": 30.0,
                                  "action": "ilk dönem memnuniyet araması",
                                  "expected_saved_value": 50.0}])
        pool_high = _make_pool([{"MonthlyCharges": 150.0,
                                  "action": "ilk dönem memnuniyet araması",
                                  "expected_saved_value": 50.0}])
        cost_low  = optimizer.assign_offer_cost(pool_low)["offer_cost"].iloc[0]
        cost_high = optimizer.assign_offer_cost(pool_high)["offer_cost"].iloc[0]
        assert cost_low == pytest.approx(cost_high)

    def test_unknown_action_cost_zero(self):
        """Bilinmeyen aksiyon → maliyet = 0.0."""
        pool = _make_pool([
            {"action": "TAMAMEN_BILINMEYEN_AKSIYON",
             "MonthlyCharges": 70.0, "expected_saved_value": 100.0},
        ])
        result = optimizer.assign_offer_cost(pool)
        assert result["offer_cost"].iloc[0] == 0.0
