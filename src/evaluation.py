"""
evaluation.py
Strateji karşılaştırma ve değerlendirme modülü.

Üç stratejiyi karşılaştırır:
  1. Agent Stratejisi  — kişiselleştirilmiş aksiyon + ROI-öncelikli bütçe opt.
  2. Baseline          — herkese sabit 10 TL memnuniyet araması
  3. Risk-Only         — yalnızca yüksek riskli müşterilere %10 indirim

Karşılaştırma Adaleti Notu (tez için kritik):
  Her strateji farklı bir naif yaklaşımı temsil eder ve kasıtlı olarak
  alt sınır görevi görür. Baseline ve Risk-Only stratejiler de kendi
  içlerinde net_benefit'e göre sıralanarak bütçe kısıtı uygulanır;
  bu sayede karşılaştırma "greedy order" açısından tutarlıdır.
  Kalan fark metodolojik üstünlüğü (kişiselleştirme + ROI optimizasyonu)
  yansıtır ve tezde bu şekilde yorumlanmalıdır.

  Stratejilerin kasıtlı kısıtları:
    - Baseline : uplift ve maliyet sabit — kişiselleştirme yok
    - Risk-Only : yalnızca yüksek riskli müşteriler, sabit indirim
    - Agent     : kişiselleştirilmiş, ROI-öncelikli greedy seçim

Precision@K:
  Üç strateji için de hesaplanır; seçilen müşteriler arasında gerçekten
  yüksek churn olasılığı (≥0.5) olanların oranını ölçer.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StrategyEvaluator:

    # -----------------------------------------------------------------------
    # Yardımcı: Precision@K
    # -----------------------------------------------------------------------

    @staticmethod
    def _precision_at_k(df: pd.DataFrame, proba_col: str = "churn_proba") -> float | None:
        """
        Seçilen müşterilerde churn_proba >= 0.5 olanların oranı.
        Operasyonel hassasiyet (Precision@K) göstergesi.
        """
        if proba_col not in df.columns or len(df) == 0:
            return None
        return round(float((df[proba_col] >= 0.5).mean()), 3)

    @staticmethod
    def _empty_result(strategy: str) -> dict:
        return {
            "strategy":        strategy,
            "selected_count":  0,
            "total_cost":      0.0,
            "expected_saved":  0.0,
            "net_benefit":     0.0,
            "avg_roi":         0.0,
            "cost_efficiency": 0.0,
            "precision_at_k":  None,
        }

    @staticmethod
    def _apply_budget_greedy(
        df:         pd.DataFrame,
        cost_col:   str,
        benefit_col: str,
        max_budget: float,
    ) -> pd.DataFrame:
        """
        net_benefit'e göre sıralanmış DataFrame'e bütçe kısıtı uygular.

        Tüm stratejiler bu ortak yardımcıyı kullanır; sıralama tutarlılığı
        sağlanarak karşılaştırma adaleti korunur.
        """
        df = df.sort_values(benefit_col, ascending=False).reset_index(drop=True)
        cumcost = df[cost_col].cumsum()
        return df[cumcost <= max_budget].copy()

    # -----------------------------------------------------------------------
    # 1. Agent stratejisi
    #    Havuz: RetentionOptimizer.select_by_constraints() çıktısı
    # -----------------------------------------------------------------------

    def evaluate_agent_strategy(self, optimized: pd.DataFrame) -> dict:
        """
        Agent stratejisi değerlendirmesi.
        Girdi olarak RetentionOptimizer.select_by_constraints() çıktısı alınır.
        Uplift oranları müşteri segmentine göre dinamik atanmıştır (RetentionAgent).
        """
        if len(optimized) == 0:
            logger.info("Agent stratejisi: seçilen müşteri yok.")
            return self._empty_result("Yapay Zekâ Destekli Akıllı Yaklaşım")

        total_cost     = optimized["offer_cost"].sum()
        expected_saved = optimized["expected_saved_value"].sum()
        net_benefit    = optimized["net_benefit"].sum()

        result = {
            "strategy":        "Yapay Zekâ Destekli Akıllı Yaklaşım",
            "selected_count":  len(optimized),
            "total_cost":      round(total_cost, 2),
            "expected_saved":  round(expected_saved, 2),
            "net_benefit":     round(net_benefit, 2),
            "avg_roi":         round(float(optimized["roi"].mean()), 3),
            "cost_efficiency": round(
                expected_saved / total_cost if total_cost > 0 else 0.0, 3
            ),
            "precision_at_k":  self._precision_at_k(optimized),
        }
        logger.info(
            "Agent stratejisi: %d müşteri, net_benefit=%.2f TL, ROI=%.3f",
            result["selected_count"], result["net_benefit"], result["avg_roi"],
        )
        return result

    # -----------------------------------------------------------------------
    # 2. Baseline stratejisi
    #    Havuz: aday havuzundaki TÜM müşteriler (yüksek + orta risk)
    #    Aksiyon: herkese sabit 10 TL memnuniyet araması, sabit uplift=0.08
    #
    #    Kasıtlı kısıt: Uplift ve maliyet kişiselleştirilmemiştir.
    #    Bu strateji, agent sisteminin üstünlüğünü kanıtlamak için
    #    naif alt sınır görevi görür.
    # -----------------------------------------------------------------------

    def evaluate_baseline_strategy(
        self,
        candidate_pool: pd.DataFrame,
        max_budget:     float,
    ) -> dict:
        """
        Baseline stratejisi: tüm aday havuzuna sabit 10 TL temas.
        Sıralama: net_benefit_bl ↓ (greedy tutarlılığı için).
        """
        df = candidate_pool.copy()
        df["offer_cost_bl"]       = 10.0
        df["retention_uplift_bl"] = 0.08
        df["expected_saved_bl"]   = df["expected_loss"] * 0.08
        df["net_benefit_bl"]      = df["expected_saved_bl"] - df["offer_cost_bl"]

        affordable = df[df["net_benefit_bl"] > 0].copy()
        affordable = self._apply_budget_greedy(
            affordable, "offer_cost_bl", "net_benefit_bl", max_budget
        )

        if len(affordable) == 0:
            logger.info("Baseline stratejisi: bütçe kısıtı altında seçilecek müşteri yok.")
            return self._empty_result("Geleneksel Toplu Yaklaşım")

        total_cost     = affordable["offer_cost_bl"].sum()
        expected_saved = affordable["expected_saved_bl"].sum()
        net_benefit    = affordable["net_benefit_bl"].sum()
        avg_roi        = float(
            (affordable["net_benefit_bl"] / affordable["offer_cost_bl"]).mean()
        )

        result = {
            "strategy":        "Geleneksel Toplu Yaklaşım",
            "selected_count":  len(affordable),
            "total_cost":      round(total_cost, 2),
            "expected_saved":  round(expected_saved, 2),
            "net_benefit":     round(net_benefit, 2),
            "avg_roi":         round(avg_roi, 3),
            "cost_efficiency": round(
                expected_saved / total_cost if total_cost > 0 else 0.0, 3
            ),
            "precision_at_k":  self._precision_at_k(affordable),
        }
        logger.info(
            "Baseline stratejisi: %d müşteri, net_benefit=%.2f TL, ROI=%.3f",
            result["selected_count"], result["net_benefit"], result["avg_roi"],
        )
        return result

    # -----------------------------------------------------------------------
    # 3. Risk-Only stratejisi
    #    Havuz: yalnızca "Yüksek" riskli müşteriler
    #    Aksiyon: sabit %10 indirim, sabit uplift=0.20
    #
    #    Kasıtlı kısıt: Orta riskli müşterileri kapsamaz ve kişiselleştirme
    #    yoktur. Tezde bu fark açıkça belirtilmelidir.
    # -----------------------------------------------------------------------

    def evaluate_risk_only_strategy(
        self,
        candidate_pool: pd.DataFrame,
        max_budget:     float,
    ) -> dict:
        """
        Risk-Only stratejisi: yalnızca yüksek riskli müşterilere %10 indirim.
        Sıralama: net_benefit_ro ↓ (greedy tutarlılığı için).
        """
        df = candidate_pool[candidate_pool["risk_level"] == "Yüksek"].copy()

        if len(df) == 0:
            logger.info("Risk-Only stratejisi: yüksek riskli müşteri bulunamadı.")
            return self._empty_result("Risk Bazlı Sabit Aksiyon")

        df["offer_cost_ro"]       = df["MonthlyCharges"] * 0.10
        df["retention_uplift_ro"] = 0.20
        df["expected_saved_ro"]   = df["expected_loss"] * 0.20
        df["net_benefit_ro"]      = df["expected_saved_ro"] - df["offer_cost_ro"]

        df = df[df["net_benefit_ro"] > 0].copy()
        df = self._apply_budget_greedy(
            df, "offer_cost_ro", "net_benefit_ro", max_budget
        )

        if len(df) == 0:
            logger.info("Risk-Only stratejisi: bütçe kısıtı altında seçilecek müşteri yok.")
            return self._empty_result("Risk Bazlı Sabit Aksiyon")

        total_cost     = df["offer_cost_ro"].sum()
        expected_saved = df["expected_saved_ro"].sum()
        net_benefit    = df["net_benefit_ro"].sum()
        avg_roi        = float(
            (df["net_benefit_ro"] / df["offer_cost_ro"]).mean()
        )

        result = {
            "strategy":        "Risk Bazlı Sabit Aksiyon",
            "selected_count":  len(df),
            "total_cost":      round(total_cost, 2),
            "expected_saved":  round(expected_saved, 2),
            "net_benefit":     round(net_benefit, 2),
            "avg_roi":         round(avg_roi, 3),
            "cost_efficiency": round(
                expected_saved / total_cost if total_cost > 0 else 0.0, 3
            ),
            "precision_at_k":  self._precision_at_k(df),
        }
        logger.info(
            "Risk-Only stratejisi: %d müşteri, net_benefit=%.2f TL, ROI=%.3f",
            result["selected_count"], result["net_benefit"], result["avg_roi"],
        )
        return result

    # -----------------------------------------------------------------------
    # Tüm stratejileri karşılaştır
    # -----------------------------------------------------------------------

    def compare_all(
        self,
        optimized:      pd.DataFrame,
        candidate_pool: pd.DataFrame,
        max_budget:     float,
    ) -> pd.DataFrame:
        """
        Üç stratejiyi tek bir DataFrame'de karşılaştırır.

        Strateji havuzları ve kasıtlı kısıtları:
          Agent     : RetentionOptimizer çıktısı (kişiselleştirilmiş, ROI-opt.)
          Baseline  : Tüm aday havuzu (kişiselleştirilmemiş, sabit uplift)
          Risk-Only : Yalnızca yüksek riskli müşteriler (kişiselleştirilmemiş)
        """
        logger.info(
            "Strateji karşılaştırması başlatıldı — bütçe: %.2f TL", max_budget
        )
        results = [
            self.evaluate_agent_strategy(optimized),
            self.evaluate_baseline_strategy(candidate_pool, max_budget),
            self.evaluate_risk_only_strategy(candidate_pool, max_budget),
        ]
        return pd.DataFrame(results)

    # -----------------------------------------------------------------------
    # Agent üstünlük özeti
    # -----------------------------------------------------------------------

    def agent_advantage_summary(self, comparison_df: pd.DataFrame) -> dict:
        """
        Agent stratejisinin diğerlerine göre net fayda artışını hesaplar.

        Dönen sözlük:
          vs_baseline_pct  : Baseline'a göre net fayda artışı (%)
          vs_risk_only_pct : Risk-Only'e göre net fayda artışı (%)
          agent_is_best    : Agent en yüksek net faydayı sağlıyor mu?
        """
        def _get_nb(strategy_substr: str) -> float:
            row = comparison_df[
                comparison_df["strategy"].str.contains(strategy_substr, na=False)
            ]
            return float(row["net_benefit"].values[0]) if len(row) > 0 else 0.0

        agent_nb = _get_nb("Yapay Zekâ")
        bl_nb    = _get_nb("Geleneksel")
        ro_nb    = _get_nb("Risk Bazlı")

        def pct_uplift(a: float, b: float) -> float:
            return round((a - b) / max(abs(b), 1) * 100, 1)

        summary = {
            "vs_baseline_pct":  pct_uplift(agent_nb, bl_nb),
            "vs_risk_only_pct": pct_uplift(agent_nb, ro_nb),
            "agent_is_best":    agent_nb >= max(bl_nb, ro_nb),
        }
        logger.info(
            "Agent üstünlüğü — Baseline: +%.1f%%, Risk-Only: +%.1f%%, En iyi: %s",
            summary["vs_baseline_pct"],
            summary["vs_risk_only_pct"],
            summary["agent_is_best"],
        )
        return summary


# ===========================================================================
# İstatistiksel A/B Testi Simülatörü
#
# Akademik bağlam (tez için):
#   Gerçek kampanya verisi olmadan kalibre edilmiş sentetik A/B grubu
#   üretilir. "Yapay Zekâ Destekli Kişiselleştirilmiş Strateji" (Grup A)
#   ile "Geleneksel Genel Strateji" (Grup B) karşılaştırılır.
#
#   İstatistiksel çerçeve:
#     - H₀: μ_A = μ_B  (AI ve geleneksel stratejinin ortalama net faydası eşit)
#     - H₁: μ_A > μ_B  (AI stratejisi daha yüksek net fayda sağlar)
#     - Test: Welch t-testi (eşit varyans varsayımı gereksiz)
#     - CI : Bootstrap persentil (%95, 2000 iterasyon)
#     - Güç : Welch t-dağılımı üzerinden non-central t-dağılımı yaklaşımı
#     - Etki büyüklüğü: Cohen's d
#
#   Kaynak:
#     Welch, B. L. (1947). Biometrika, 34(1–2), 28–35.
#     Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
#     Efron & Hastie (2016). Computer Age Statistical Inference, Ch. 11.
# ===========================================================================

class ABTestSimulator:
    """
    Sentetik A/B testi simülatörü.

    Grup A — AI Destekli Kişiselleştirilmiş Strateji:
        Gerçek optimizasyon çıktısından alınan net_benefit değerleri.

    Grup B — Geleneksel Genel Strateji:
        Aynı müşteri havuzuna sabit maliyet (10 TL) ve sabit uplift (%8)
        uygulanarak hesaplanan sentetik net_benefit değerleri.

    Bootstrap CI, bootstrap dağılımının persentil yöntemiyle hesaplanır
    (teorik normallik varsayımı gerekmez; küçük örnekler için daha sağlam).
    """

    N_BOOTSTRAP      = 2_000   # Bootstrap iterasyon sayısı
    ALPHA            = 0.05    # Anlamlılık düzeyi (tek-kuyruk yorumlanır)
    BASELINE_COST    = 10.0    # Geleneksel strateji: sabit temas maliyeti (TL)
    BASELINE_UPLIFT  = 0.08    # Geleneksel strateji: sabit elde tutma uplift'i

    def run_simulation(
        self,
        optimized:      pd.DataFrame,
        candidate_pool: pd.DataFrame,
        seed:           int = 42,
    ) -> dict:
        """
        AI ve geleneksel stratejiyi karşılaştıran istatistiksel A/B testi çalıştırır.

        Parametreler
        ------------
        optimized      : RetentionOptimizer çıktısı (AI Grubu — Grup A)
        candidate_pool : Tüm aday havuzu (Geleneksel Grup B senteği için taban)
        seed           : Tekrarlanabilirlik için rastgele tohum

        Döndürür
        --------
        dict : test istatistikleri, CI, güç ve yorum
        """
        rng = np.random.default_rng(seed)

        # --- Grup A: AI Destekli Strateji (gerçek optimizasyon çıktısı) ----
        ai_benefits = optimized["net_benefit"].values.astype(float)

        # --- Grup B: Geleneksel Strateji (sentetik hesaplama) ---------------
        # Aynı müşteri havuzuna sabit maliyet ve sabit uplift uygulanır.
        # Bu yapay alt sınır, kişiselleştirmenin katkısını izole eder.
        trad_df = candidate_pool.copy()
        trad_df["_trad_saved"] = trad_df["expected_loss"] * self.BASELINE_UPLIFT
        trad_df["_trad_net"]   = trad_df["_trad_saved"] - self.BASELINE_COST
        # Yalnızca pozitif net faydası olan müşteriler seçilir (hâklı karşılaştırma)
        trad_benefits = trad_df[trad_df["_trad_net"] > 0]["_trad_net"].values.astype(float)

        n_ai   = len(ai_benefits)
        n_trad = len(trad_benefits)

        if n_ai < 2 or n_trad < 2:
            return {"error": "İstatistiksel test için yeterli veri yok (n < 2)."}

        # --- Welch t-testi --------------------------------------------------
        from scipy import stats as scipy_stats

        t_stat, p_value_two_sided = scipy_stats.ttest_ind(
            ai_benefits, trad_benefits, equal_var=False
        )
        # Tek-kuyruk (AI > Geleneksel) → H₁: μ_A > μ_B
        p_value = p_value_two_sided / 2 if t_stat > 0 else 1.0 - p_value_two_sided / 2

        # --- Ortalamalar ve fark -------------------------------------------
        mean_ai   = float(ai_benefits.mean())
        mean_trad = float(trad_benefits.mean())
        diff_mean = mean_ai - mean_trad

        # --- Bootstrap %95 Güven Aralığı ------------------------------------
        boot_diffs = np.empty(self.N_BOOTSTRAP)
        for i in range(self.N_BOOTSTRAP):
            sample_ai   = rng.choice(ai_benefits,   size=n_ai,   replace=True)
            sample_trad = rng.choice(trad_benefits, size=n_trad, replace=True)
            boot_diffs[i] = sample_ai.mean() - sample_trad.mean()

        ci_lower = float(np.percentile(boot_diffs, 2.5))
        ci_upper = float(np.percentile(boot_diffs, 97.5))

        # --- Cohen's d (standartlaştırılmış etki büyüklüğü) ----------------
        pooled_std = float(np.sqrt(
            (np.var(ai_benefits, ddof=1) + np.var(trad_benefits, ddof=1)) / 2
        ))
        cohens_d = diff_mean / pooled_std if pooled_std > 0 else 0.0

        # --- İstatistiksel Güç Analizi (non-central t yaklaşımı) -----------
        # Harmonic mean sample size → dengeli Welch güç tahmini
        n_harmonic = 2 / (1 / n_ai + 1 / n_trad)
        ncp        = cohens_d * np.sqrt(n_harmonic / 2)   # non-centrality parameter
        df_welch   = n_ai + n_trad - 2

        t_critical = scipy_stats.t.ppf(1 - self.ALPHA, df=df_welch)
        power      = float(1 - scipy_stats.nct.cdf(t_critical, df=df_welch, nc=ncp))

        return {
            "n_ai":                  n_ai,
            "n_traditional":         n_trad,
            "mean_ai_benefit":       round(mean_ai,    2),
            "mean_trad_benefit":     round(mean_trad,  2),
            "mean_difference":       round(diff_mean,  2),
            "t_statistic":           round(float(t_stat),  4),
            "p_value":               round(p_value,    6),
            "p_value_significant":   bool(p_value < self.ALPHA),
            "ci_95_lower":           round(ci_lower,   2),
            "ci_95_upper":           round(ci_upper,   2),
            "cohens_d":              round(cohens_d,   4),
            "effect_magnitude":      self._effect_label(cohens_d),
            "statistical_power":     round(power,      4),
            "power_adequate":        bool(power >= 0.80),
            "bootstrap_iterations":  self.N_BOOTSTRAP,
            "alpha":                 self.ALPHA,
            "interpretation":        self._interpret(p_value, power, cohens_d, diff_mean),
        }

    @staticmethod
    def _effect_label(d: float) -> str:
        """Cohen's d büyüklüğünü sözel etiketle açıklar (Cohen, 1988)."""
        abs_d = abs(d)
        if abs_d >= 0.80:
            return "Büyük etki (d ≥ 0.80)"
        if abs_d >= 0.50:
            return "Orta etki (0.50 ≤ d < 0.80)"
        if abs_d >= 0.20:
            return "Küçük etki (0.20 ≤ d < 0.50)"
        return "İhmal edilebilir etki (d < 0.20)"

    @staticmethod
    def _interpret(p: float, power: float, d: float, diff: float) -> str:
        """İstatistiksel bulgulardan akademik yorum metni üretir."""
        sig  = p < 0.05
        lines = []

        if sig and diff > 0:
            lines.append(
                f"H₀ reddedildi (p = {p:.4f} < 0.05): AI destekli stratejinin ortalama net faydası "
                f"geleneksel stratejiden istatistiksel olarak anlamlı biçimde yüksektir. "
                f"Bu bulgu tesadüfi değil, metodolojik üstünlüğe işaret etmektedir."
            )
        elif sig and diff <= 0:
            lines.append(
                f"H₀ reddedildi (p = {p:.4f} < 0.05), ancak fark geleneksel strateji lehinedir. "
                f"Bütçe veya havuz kısıtları gözden geçirilmelidir."
            )
        else:
            lines.append(
                f"H₀ reddedilemedi (p = {p:.4f} ≥ 0.05): İki strateji arasında "
                f"istatistiksel olarak anlamlı bir fark saptanamamıştır. "
                f"Daha büyük örneklem veya daha geniş bütçeyle test tekrarlanmalıdır."
            )

        if power >= 0.80:
            lines.append(
                f"İstatistiksel güç yeterli (β = {power:.2f} ≥ 0.80): "
                f"Mevcut örneklem boyutu gerçek bir etkiyi tespit etmek için yeterlidir."
            )
        else:
            lines.append(
                f"İstatistiksel güç düşük (β = {power:.2f} < 0.80): "
                f"Daha fazla müşteri gözlemine ihtiyaç vardır (tip-II hata riski yüksek)."
            )

        return " ".join(lines)
