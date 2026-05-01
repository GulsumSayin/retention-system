"""
optimization.py
Bütçe kısıtı altında müşteri aksiyon optimizasyonu.

Algoritma Seçimi ve Teorik Gerekçe:
------------------------------------
Müşteri seçim problemi bir 0/1 Knapsack problemidir ve NP-hard olduğu
bilinmektedir (Karp, 1972). Tam çözüm için dinamik programlama O(n×W)
karmaşıklığına sahiptir ve büyük müşteri havuzlarında pratik değildir.

Greedy Yaklaşım (ROI-öncelikli):
  Sıralama kriteri: ROI = net_benefit / offer_cost (değer/ağırlık oranı)
  Bu sıralama, Fractional Knapsack'in optimal çözümüdür (Dantzig, 1957);
  0/1 Knapsack için optimalliği garanti etmez ancak standart greedy
  yaklaşımlarının en iyi bilinen heuristiğidir.

  Önceki sürümdeki net_benefit-öncelikli sıralamanın sorunu:
    Pahalı ama yüksek net değerli müşteriler, birden fazla düşük-maliyetli
    yüksek-ROI müşterisini bütçe kısıtıyla dışarıda bırakabilir.
  ROI-öncelikli sıralama bütçe verimliliğini maksimize eder.

  İkincil ve üçüncül kriter: net_benefit → priority_score
  (eşit ROI durumunda mutlak değeri büyük olan tercih edilir).

Maliyet Hesaplama:
  Tüm aksiyon maliyetleri ACTION_REGISTRY'den okunur (tek kaynak).
  Bilinmeyen aksiyon → 0.0 maliyet → offer_cost filtresiyle elendiğinden
  sessiz hata yerine logger.warning tetiklenir.

Gelecek Çalışma:
  - Dinamik programlama veya LP relaxation ile greedy karşılaştırması
  - Integer Linear Programming (ILP) formülasyonu
"""

import logging

import numpy as np
import pandas as pd

from action_registry import get_action_spec

logger = logging.getLogger(__name__)


class RetentionOptimizer:
    """
    Bütçe (ve isteğe bağlı müşteri sayısı) kısıtı altında
    net faydayı maksimize eden müşteri kümesini seçer.
    """

    # -----------------------------------------------------------------------
    # 1. Aksiyon maliyeti hesaplama
    # -----------------------------------------------------------------------

    def assign_offer_cost(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Her müşteriye atanan aksiyonun tahmini maliyetini ACTION_REGISTRY'den
        hesaplar.

        Maliyet türleri (bkz. action_registry.py):
          - "rate"  : aylık ücretin belirli bir oranı (indirim + operasyon)
          - "fixed" : sabit TL (temas / bilgilendirme aksiyonları)

        Net fayda formülü:
          net_benefit = expected_saved_value − offer_cost

        ROI formülü:
          roi = net_benefit / offer_cost   (offer_cost > 0 iken)
        """
        result = df.copy()

        def cost_fn(row: pd.Series) -> float:
            action  = row.get("action", "")
            monthly = row.get("MonthlyCharges", 0.0)
            spec    = get_action_spec(action)

            if spec is None:
                logger.warning(
                    "assign_offer_cost: Bilinmeyen aksiyon '%s' — maliyet 0.0 atandı. "
                    "action_registry.py'yi kontrol edin.",
                    action,
                )
                return 0.0

            if spec["cost_type"] == "rate":
                return float(monthly) * spec["cost_value"]
            return spec["cost_value"]

        result["offer_cost"]  = result.apply(cost_fn, axis=1)
        result["net_benefit"] = result["expected_saved_value"] - result["offer_cost"]
        result["roi"] = np.where(
            result["offer_cost"] > 0,
            result["net_benefit"] / result["offer_cost"],
            0.0,
        )

        return result

    # -----------------------------------------------------------------------
    # 2. Kısıtlı seçim algoritması (Greedy — ROI öncelikli)
    # -----------------------------------------------------------------------

    def select_by_constraints(
        self,
        df:            pd.DataFrame,
        max_budget:    float = 2000.0,
        max_customers: int | None = None,
    ) -> pd.DataFrame:
        """
        Bütçe (ve isteğe bağlı müşteri sayısı) kısıtı altında
        net faydayı maksimize eden müşteri kümesini seçer.

        Algoritma: Greedy — ROI ↓ → net_benefit ↓ → priority_score ↓
        ---------------------------------------------------------------
        ROI'ye göre sıralama, Fractional Knapsack optimal çözümüdür
        (Dantzig, 1957) ve 0/1 Knapsack için standart greedy heuristiğidir.
        Net değere göre birincil sıralama yapmak, bütçeyi pahalı müşterilerle
        tıkayıp daha fazla yüksek-ROI müşteriye ulaşmayı engelleyebilir.

        Ön filtreler:
          1. "müdahale gerekmiyor" aksiyonları hariç tutulur.
          2. offer_cost = 0 olan satırlar (bilinmeyen aksiyon) hariç tutulur.
          3. net_benefit ≤ 0 olan müşteriler hariç tutulur
             (kurtarma değeri maliyeti karşılamıyor).

        Parametreler
        ------------
        df             : Aday havuzu DataFrame'i (RetentionAgent çıktısı)
        max_budget     : Maksimum kampanya bütçesi (TL)
        max_customers  : Seçilecek maksimum müşteri sayısı (None = kısıtsız)

        Döndürür
        --------
        pd.DataFrame : Seçilen müşteriler (bütçe kısıtına uyan, pozitif net faydası olan)
        """
        result = df.copy()
        result = self.assign_offer_cost(result)

        n_before       = len(result)
        result         = result[result["action"] != "müdahale gerekmiyor"].copy()
        n_after_step1  = len(result)
        result         = result[result["offer_cost"] > 0].copy()
        n_after_step2  = len(result)
        result         = result[result["net_benefit"] > 0].copy()
        n_after        = len(result)

        logger.info(
            "Optimizasyon filtresi: %d → %d müşteri "
            "(müdahale=%d, maliyet=0=%d, negatif_net=%d elendi)",
            n_before, n_after,
            n_before      - n_after_step1,
            n_after_step1 - n_after_step2,
            n_after_step2 - n_after,
        )

        # ROI öncelikli sıralama (Greedy Knapsack standardı)
        result = result.sort_values(
            ["roi", "net_benefit", "priority_score"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        # Greedy seçim
        selected_rows = []
        total_budget  = 0.0

        for _, row in result.iterrows():
            if max_customers is not None and len(selected_rows) >= max_customers:
                break
            next_budget = total_budget + row["offer_cost"]
            if next_budget <= max_budget:
                selected_rows.append(row)
                total_budget = next_budget

        if not selected_rows:
            logger.warning(
                "Optimizasyon: Bütçe (%.2f TL) ile seçilecek müşteri bulunamadı.",
                max_budget,
            )
            return pd.DataFrame(columns=result.columns.tolist())

        selected = pd.DataFrame(selected_rows).reset_index(drop=True)
        logger.info(
            "Optimizasyon tamamlandı: %d müşteri seçildi, "
            "toplam maliyet %.2f TL / %.2f TL bütçe (%.1f%% kullanım).",
            len(selected),
            total_budget,
            max_budget,
            100 * total_budget / max_budget if max_budget > 0 else 0,
        )
        return selected
