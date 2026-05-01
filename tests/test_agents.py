"""
tests/test_agents.py
RetentionAgent ve CLV formülü birim testleri.

Test Kapsamı:
  1. CLV Formülü — p² tuzağı yok; LTV churn_proba'dan bağımsız
  2. expected_loss — doğrusal E[kayıp] = p × LTV
  3. estimated_clv — raporlama iskonto doğruluğu
  4. priority_score — çift sayma yok
  5. Sözleşme tipi çarpanı (1.0 / 1.3 / 1.6)
  6. Risk seviyesi eşikleri (0.70 / 0.40 / <0.40)
  7. Uplift oranı — tenure ve churn_proba düzeltmeleri
  8. Eksik sütunlar için sıfır-varsayılan güvenliği
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import pandas as pd
import numpy as np

from agents import RetentionAgent, rule_based_reason


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

def _make_customer(**kwargs) -> pd.DataFrame:
    """Minimal geçerli bir müşteri DataFrame'i oluşturur."""
    defaults = {
        "churn_proba":     0.80,
        "MonthlyCharges":  75.0,
        "tenure":          12,
        "Contract":        "Month-to-month",
        "InternetService": "Fiber optic",
        "IsMonthToMonth":  1,
        "ServiceIntensity": 0.5,
        "NoProtectionFlag": 0,
        "UsesAutoPayment":  0,
        "NumServices":      3,
    }
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


agent = RetentionAgent()


# ===========================================================================
# 1. CLV Formülü Testleri
# ===========================================================================

class TestCLVFormula:

    def test_ltv_independent_of_churn_proba(self):
        """LTV, churn_proba değiştiğinde sabit kalmalı."""
        df_low  = _make_customer(churn_proba=0.10, MonthlyCharges=50.0, tenure=12, Contract="Month-to-month")
        df_high = _make_customer(churn_proba=0.90, MonthlyCharges=50.0, tenure=12, Contract="Month-to-month")
        r_low  = agent.compute_business_scores(df_low)
        r_high = agent.compute_business_scores(df_high)
        assert r_low["lifetime_value"].iloc[0] == pytest.approx(r_high["lifetime_value"].iloc[0])

    def test_ltv_formula_month_to_month(self):
        """LTV = MC × (6 + tenure × 0.5) × 1.0 (Month-to-month)."""
        mc, tenure = 50.0, 12
        expected_ltv = mc * (6 + tenure * 0.5) * 1.0
        df = _make_customer(MonthlyCharges=mc, tenure=tenure, Contract="Month-to-month")
        result = agent.compute_business_scores(df)
        assert result["lifetime_value"].iloc[0] == pytest.approx(expected_ltv)

    def test_ltv_formula_one_year_contract(self):
        """LTV = MC × (6 + tenure × 0.5) × 1.3 (One year)."""
        mc, tenure = 60.0, 24
        expected_ltv = mc * (6 + tenure * 0.5) * 1.3
        df = _make_customer(MonthlyCharges=mc, tenure=tenure, Contract="One year")
        result = agent.compute_business_scores(df)
        assert result["lifetime_value"].iloc[0] == pytest.approx(expected_ltv)

    def test_ltv_formula_two_year_contract(self):
        """LTV = MC × (6 + tenure × 0.5) × 1.6 (Two year)."""
        mc, tenure = 80.0, 36
        expected_ltv = mc * (6 + tenure * 0.5) * 1.6
        df = _make_customer(MonthlyCharges=mc, tenure=tenure, Contract="Two year")
        result = agent.compute_business_scores(df)
        assert result["lifetime_value"].iloc[0] == pytest.approx(expected_ltv)

    def test_expected_loss_is_linear(self):
        """expected_loss = churn_proba × LTV — p² terimi olmamalı."""
        df = _make_customer(churn_proba=0.5, MonthlyCharges=100.0, tenure=10)
        result = agent.compute_business_scores(df)
        ltv  = result["lifetime_value"].iloc[0]
        loss = result["expected_loss"].iloc[0]
        assert loss == pytest.approx(0.5 * ltv)

    def test_estimated_clv_discount_formula(self):
        """estimated_clv = LTV × (1 − p × 0.3)."""
        p, mc, tenure = 0.6, 70.0, 18
        df = _make_customer(churn_proba=p, MonthlyCharges=mc, tenure=tenure, Contract="Month-to-month")
        result = agent.compute_business_scores(df)
        ltv = result["lifetime_value"].iloc[0]
        expected_clv = ltv * (1 - p * 0.3)
        assert result["estimated_clv"].iloc[0] == pytest.approx(expected_clv)

    def test_estimated_clv_does_not_feed_expected_loss(self):
        """estimated_clv, expected_loss hesabında kullanılmamalı (p² tuzağı)."""
        df = _make_customer(churn_proba=0.8, MonthlyCharges=90.0, tenure=20)
        result = agent.compute_business_scores(df)
        # expected_loss = p × LTV olmalı, estimated_clv × p olmamalı
        ltv  = result["lifetime_value"].iloc[0]
        loss = result["expected_loss"].iloc[0]
        p    = 0.8
        assert loss == pytest.approx(p * ltv, rel=1e-6)
        # Yanlış formül: p × estimated_clv = p × LTV × (1 - p×0.3) → farklı sonuç
        wrong = p * result["estimated_clv"].iloc[0]
        assert abs(loss - wrong) > 1e-3  # farklı olmalı

    def test_zero_tenure_ltv(self):
        """Tenure = 0 iken LTV = MC × 6 × contract_mult."""
        mc = 50.0
        df = _make_customer(MonthlyCharges=mc, tenure=0, Contract="Month-to-month")
        result = agent.compute_business_scores(df)
        assert result["lifetime_value"].iloc[0] == pytest.approx(mc * 6 * 1.0)


# ===========================================================================
# 2. Priority Score Testleri
# ===========================================================================

class TestPriorityScore:

    def test_priority_score_formula(self):
        """priority_score = expected_loss × (1 + IsMonthToMonth) × (1 + ServiceIntensity)."""
        df = _make_customer(
            churn_proba=0.7, MonthlyCharges=80.0, tenure=6,
            Contract="Month-to-month", IsMonthToMonth=1, ServiceIntensity=0.4,
        )
        result = agent.compute_business_scores(df)
        loss     = result["expected_loss"].iloc[0]
        expected = loss * (1 + 1) * (1 + 0.4)
        assert result["priority_score"].iloc[0] == pytest.approx(expected)

    def test_priority_score_no_mtm_bonus(self):
        """IsMonthToMonth=0 ise çarpan 1 olmalı."""
        df = _make_customer(
            churn_proba=0.6, MonthlyCharges=60.0, tenure=24,
            Contract="Two year", IsMonthToMonth=0, ServiceIntensity=0.0,
        )
        result = agent.compute_business_scores(df)
        loss = result["expected_loss"].iloc[0]
        assert result["priority_score"].iloc[0] == pytest.approx(loss)

    def test_higher_risk_gives_higher_priority(self):
        """Daha yüksek churn_proba → daha yüksek priority_score (diğer her şey eşit)."""
        df_low  = _make_customer(churn_proba=0.3, MonthlyCharges=70.0, tenure=12)
        df_high = _make_customer(churn_proba=0.9, MonthlyCharges=70.0, tenure=12)
        r_low  = agent.compute_business_scores(df_low)
        r_high = agent.compute_business_scores(df_high)
        assert r_high["priority_score"].iloc[0] > r_low["priority_score"].iloc[0]

    def test_run_sorts_by_priority_desc(self):
        """run() priority_score'a göre azalan sıralama yapmalı."""
        rows = [
            {"churn_proba": 0.3, "MonthlyCharges": 40.0, "tenure": 6,
             "Contract": "Month-to-month", "IsMonthToMonth": 1, "ServiceIntensity": 0.0},
            {"churn_proba": 0.9, "MonthlyCharges": 90.0, "tenure": 24,
             "Contract": "Month-to-month", "IsMonthToMonth": 1, "ServiceIntensity": 1.0},
            {"churn_proba": 0.6, "MonthlyCharges": 65.0, "tenure": 12,
             "Contract": "One year", "IsMonthToMonth": 0, "ServiceIntensity": 0.5},
        ]
        df = pd.DataFrame(rows)
        result = agent.run(df)
        scores = result["priority_score"].tolist()
        assert scores == sorted(scores, reverse=True)


# ===========================================================================
# 3. Risk Seviyesi Testleri
# ===========================================================================

class TestRiskLevel:

    def test_high_risk_threshold(self):
        """churn_proba >= 0.70 → 'Yüksek'."""
        for p in [0.70, 0.85, 1.0]:
            df = _make_customer(churn_proba=p)
            result = agent.assign_risk_level(agent.compute_business_scores(df))
            assert result["risk_level"].iloc[0] == "Yüksek", f"p={p}"

    def test_medium_risk_threshold(self):
        """0.40 <= churn_proba < 0.70 → 'Orta'."""
        for p in [0.40, 0.55, 0.699]:
            df = _make_customer(churn_proba=p)
            result = agent.assign_risk_level(agent.compute_business_scores(df))
            assert result["risk_level"].iloc[0] == "Orta", f"p={p}"

    def test_low_risk_threshold(self):
        """churn_proba < 0.40 → 'Düşük'."""
        for p in [0.0, 0.20, 0.399]:
            df = _make_customer(churn_proba=p)
            result = agent.assign_risk_level(agent.compute_business_scores(df))
            assert result["risk_level"].iloc[0] == "Düşük", f"p={p}"

    def test_boundary_exactly_0_70(self):
        """p = 0.70 tam sınırda 'Yüksek' olmalı."""
        df = _make_customer(churn_proba=0.70)
        result = agent.assign_risk_level(agent.compute_business_scores(df))
        assert result["risk_level"].iloc[0] == "Yüksek"

    def test_boundary_exactly_0_40(self):
        """p = 0.40 tam sınırda 'Orta' olmalı."""
        df = _make_customer(churn_proba=0.40)
        result = agent.assign_risk_level(agent.compute_business_scores(df))
        assert result["risk_level"].iloc[0] == "Orta"


# ===========================================================================
# 4. Uplift Oranı Testleri
# ===========================================================================

class TestUpliftRate:

    def test_unknown_action_returns_zero(self):
        """Bilinmeyen aksiyon → uplift = 0.0."""
        row = pd.Series({"action": "BILINMEYEN_AKSIYON", "tenure": 12, "churn_proba": 0.5})
        assert agent.assign_uplift_rate(row) == 0.0

    def test_long_tenure_increases_uplift(self):
        """tenure > 24 → base_uplift × 1.15 (min(, 0.50))."""
        action = "%10 indirim + 12 ay taahhüt"
        row_short = pd.Series({"action": action, "tenure": 12, "churn_proba": 0.5})
        row_long  = pd.Series({"action": action, "tenure": 30, "churn_proba": 0.5})
        assert agent.assign_uplift_rate(row_long) > agent.assign_uplift_rate(row_short)

    def test_short_tenure_decreases_uplift(self):
        """tenure < 6 → base_uplift × 0.85."""
        action = "müşteri temsilcisi ile öncelikli temas"
        row_normal = pd.Series({"action": action, "tenure": 12, "churn_proba": 0.5})
        row_short  = pd.Series({"action": action, "tenure": 3,  "churn_proba": 0.5})
        assert agent.assign_uplift_rate(row_short) < agent.assign_uplift_rate(row_normal)

    def test_extreme_churn_reduces_uplift(self):
        """churn_proba > 0.85 → base × 0.90."""
        action = "müşteri temsilcisi ile öncelikli temas"
        row_normal = pd.Series({"action": action, "tenure": 12, "churn_proba": 0.5})
        row_extreme = pd.Series({"action": action, "tenure": 12, "churn_proba": 0.9})
        assert agent.assign_uplift_rate(row_extreme) < agent.assign_uplift_rate(row_normal)

    def test_uplift_capped_at_0_50(self):
        """Uplift asla 0.50'yi aşmamalı."""
        for action_name in [
            "%10 indirim + 12 ay taahhüt",
            "fiber paket sadakat indirimi + yıllık sözleşme",
        ]:
            row = pd.Series({"action": action_name, "tenure": 60, "churn_proba": 0.5})
            assert agent.assign_uplift_rate(row) <= 0.50


# ===========================================================================
# 5. Eksik Sütun Güvenliği
# ===========================================================================

class TestMissingColumnSafety:

    def test_compute_scores_without_optional_cols(self):
        """Opsiyonel sütunlar olmadan compute_business_scores çalışmalı."""
        df = pd.DataFrame([{
            "churn_proba":    0.5,
            "MonthlyCharges": 50.0,
            "tenure":         6,
            "Contract":       "Month-to-month",
        }])
        result = agent.compute_business_scores(df)
        assert "lifetime_value" in result.columns
        assert "expected_loss" in result.columns
        assert "priority_score" in result.columns

    def test_compute_scores_without_tenure(self):
        """tenure sütunu yoksa sıfır varsayılanıyla çalışmalı."""
        df = pd.DataFrame([{
            "churn_proba":    0.7,
            "MonthlyCharges": 60.0,
            "Contract":       "Month-to-month",
        }])
        result = agent.compute_business_scores(df)
        # tenure=0 → LTV = MC × 6
        expected_ltv = 60.0 * 6 * 1.0
        assert result["lifetime_value"].iloc[0] == pytest.approx(expected_ltv)


# ===========================================================================
# 6. rule_based_reason Testleri
# ===========================================================================

class TestRuleBasedReason:

    def test_month_to_month_mentioned(self):
        row = pd.Series({"Contract": "Month-to-month", "MonthlyCharges": 60.0,
                          "tenure": 20, "UsesAutoPayment": 1,
                          "NoProtectionFlag": 0, "InternetService": "DSL",
                          "NumServices": 5})
        reason = rule_based_reason(row)
        assert "aylık sözleşme" in reason

    def test_balanced_profile_no_risk(self):
        """Risk faktörü yoksa 'Dengeli müşteri profili' döner."""
        row = pd.Series({"Contract": "Two year", "MonthlyCharges": 50.0,
                          "tenure": 36, "UsesAutoPayment": 1,
                          "NoProtectionFlag": 0, "InternetService": "DSL",
                          "NumServices": 6})
        reason = rule_based_reason(row)
        assert "Dengeli müşteri profili" in reason

    def test_high_charges_mentioned(self):
        row = pd.Series({"Contract": "Two year", "MonthlyCharges": 85.0,
                          "tenure": 36, "UsesAutoPayment": 1,
                          "NoProtectionFlag": 0, "InternetService": "DSL",
                          "NumServices": 5})
        reason = rule_based_reason(row)
        assert "yüksek aylık ücret" in reason
