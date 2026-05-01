"""
agents.py
Kural tabanlı müşteri tutundurma ajan katmanı.

Formüller ve Matematiksel Yapı:
-------------------------------
1. Lifetime Value (LTV) — churn etkisinden bağımsız, gerçek potansiyel değer:
     LTV = MonthlyCharges × (6 + tenure × 0.5) × contract_mult
   Katsayılar iş bilgisine dayalı varsayımsaldır (tezde açıkça belirtilmiştir).
   Validasyon önerisi: survival analysis (Kaplan-Meier) veya BG/NBD modeli.

2. Beklenen Kayıp (expected_loss) — temiz olasılık × değer formülü:
     expected_loss = churn_proba × LTV
   p² tuzağından kaçınmak için LTV, churn_proba'dan bağımsız tutulur.

3. Tahmini CLV (estimated_clv) — olasılıkla iskonto edilmiş gerçek CLV:
     estimated_clv = LTV × (1 − churn_proba × 0.3)
   Bu değer raporlama ve görselleştirme içindir; expected_loss hesabında kullanılmaz.

4. Öncelik Skoru (priority_score) — finansal risk + operasyonel aciliyet:
     priority_score = expected_loss × (1 + IsMonthToMonth) × (1 + ServiceIntensity)
   expected_loss tenure ve sözleşme tipini LTV üzerinden zaten kodlar;
   IsMonthToMonth müşterinin sistemi bırakma kolaylığını, ServiceIntensity ise
   ürün bağlılığının kırılganlığını yakalamak için eklenir.

5. Beklenen Kurtarma Değeri:
     expected_saved_value = expected_loss × retention_uplift

Tüm uplift oranları varsayımsaldır; gerçek kampanya verisiyle A/B testi veya
Difference-in-Differences yöntemiyle kalibre edilmelidir (bkz. action_registry.py).

SHAP Entegrasyonu:
  - generate_reason() önce ShapService.explain_row() kullanır (post-hoc XAI).
  - ShapService fit edilmemişse kural tabanlı fallback devreye girer.
  - Fallback mantığı bu modülde _rule_based_reason() olarak tanımlıdır;
    shap_service.py bu fonksiyonu import eder (DRY — tek kaynak).
  - set_shap_service() ile dışarıdan enjekte edilir (dependency injection).
"""

import logging
from typing import Optional

import pandas as pd

from action_registry import ACTIONS, ActionSpec, get_action_spec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kural tabanlı gerekçe — tek kaynak (shap_service.py tarafından da import edilir)
# ---------------------------------------------------------------------------

def rule_based_reason(row: pd.Series) -> str:
    """
    SHAP mevcut olmadığında iş odaklı, müşteri temsilcisine yönelik
    Türkçe gerekçe üretir.

    Bu fonksiyon hem agents.py hem de shap_service.py tarafından kullanılır;
    DRY prensibi gereği tek bir noktada tanımlanmıştır.
    """
    reasons: list[str] = []

    if row.get("Contract", "") == "Month-to-month":
        reasons.append("aylık sözleşmede olduğu için ayrılması çok kolay")
    if row.get("MonthlyCharges", 0) >= 80:
        reasons.append("yüksek fatura ödüyor ancak bağlılığı düşük")
    if row.get("tenure", 999) <= 12:
        reasons.append("henüz yeni müşteri, bağlılık henüz oluşmamış")
    if row.get("UsesAutoPayment", 1) == 0:
        reasons.append("otomatik ödeme kurulmamış, her ay ödeme sürtüşmesi yaşıyor")
    if row.get("NoProtectionFlag", 0) >= 2:
        reasons.append("güvenlik ve teknik destek paketleri eksik, ek değer sunulamıyor")
    if (
        row.get("InternetService", "") == "Fiber optic"
        and row.get("MonthlyCharges", 0) >= 85
    ):
        reasons.append("fiber internet kullanıyor, yüksek fatura nedeniyle alternatif arayışında olabilir")
    if row.get("NumServices", 10) <= 2:
        reasons.append("çok az hizmet kullanıyor, platforma bağlılığı zayıf")

    if not reasons:
        return "Müşteri dengeli bir profilde görünüyor; öngörülen risk faktörü tespit edilmedi."

    # İlk harfi büyüt, nokta ile bitir
    text = "; ".join(reasons)
    return text[0].upper() + text[1:] + "."


# ===========================================================================
# RetentionAgent
# ===========================================================================

class RetentionAgent:
    """
    Kural tabanlı müşteri tutundurma ajanı.

    Sorumluluklar:
      1. İş puanları hesaplama (LTV, expected_loss, estimated_clv, priority_score)
      2. Risk seviyesi atama
      3. Kişiselleştirilmiş aksiyon paketi önerisi (ACTION_REGISTRY'den)
      4. Aksiyon gerekçesi üretimi (SHAP öncelikli, kural tabanlı fallback)
      5. Dinamik uplift oranı atama
    """

    def __init__(self) -> None:
        self._shap_svc    = None
        self._feature_cols: list = []

    # -----------------------------------------------------------------------
    # SHAP enjeksiyonu
    # -----------------------------------------------------------------------

    def set_shap_service(self, shap_svc, feature_cols: list) -> None:
        """
        Fit edilmiş ShapService ve feature listesini enjekte eder.
        app.py'de model yüklendikten hemen sonra çağrılır.
        """
        self._shap_svc     = shap_svc
        self._feature_cols = feature_cols

    # -----------------------------------------------------------------------
    # 1. İş puanları
    # -----------------------------------------------------------------------

    def compute_business_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Her müşteri için LTV, expected_loss, estimated_clv ve priority_score
        hesaplar.

        Matematiksel ayrım:
          - LTV             : churn_proba'dan bağımsız gerçek potansiyel değer
          - expected_loss   : E[kayıp] = churn_proba × LTV  (doğrusal, p² yok)
          - estimated_clv   : Riske göre iskonto edilmiş CLV (raporlama için)
          - priority_score  : Finansal risk + operasyonel aciliyet bileşeni
        """
        result = df.copy()

        for col in ["IsMonthToMonth", "ServiceIntensity", "NoProtectionFlag", "UsesAutoPayment"]:
            if col not in result.columns:
                result[col] = 0
        if "tenure" not in result.columns:
            result["tenure"] = 0

        # Sözleşme tipi çarpanı
        contract_mult = pd.Series(1.0, index=result.index)
        if "Contract" in result.columns:
            contract_mult = result["Contract"].map(
                {"Month-to-month": 1.0, "One year": 1.3, "Two year": 1.6}
            ).fillna(1.0)

        # LTV: churn_proba'dan bağımsız, gerçek müşteri değeri
        result["lifetime_value"] = (
            result["MonthlyCharges"]
            * (6 + result["tenure"] * 0.5)
            * contract_mult
        )

        # expected_loss: doğrusal E[kayıp] = p × LTV  (p² tuzağı yok)
        result["expected_loss"] = result["churn_proba"] * result["lifetime_value"]

        # estimated_clv: olasılıkla iskonto edilmiş CLV (görselleştirme / raporlama)
        result["estimated_clv"] = (
            result["lifetime_value"] * (1 - result["churn_proba"] * 0.3)
        )

        # priority_score: finansal risk + operasyonel aciliyet
        # expected_loss zaten tenure ve sözleşme tipini LTV üzerinden kodlar;
        # IsMonthToMonth ve ServiceIntensity bağımsız operasyonel boyutları ekler.
        result["priority_score"] = (
            result["expected_loss"]
            * (1 + result["IsMonthToMonth"])
            * (1 + result["ServiceIntensity"])
        )

        result["ranking_score"] = result["churn_proba"]
        return result

    # -----------------------------------------------------------------------
    # 2. Risk seviyesi
    # -----------------------------------------------------------------------

    def assign_risk_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """churn_proba eşiklerine göre Yüksek / Orta / Düşük risk atar."""
        result = df.copy()

        def _risk(p: float) -> str:
            if p >= 0.70:
                return "Yüksek"
            if p >= 0.40:
                return "Orta"
            return "Düşük"

        result["risk_level"] = result["churn_proba"].apply(_risk)
        return result

    # -----------------------------------------------------------------------
    # 3. Aksiyon paketi
    # -----------------------------------------------------------------------

    def recommend_action_bundle(self, row: pd.Series) -> dict:
        """
        Müşteri profiline göre kural tabanlı aksiyon detayını seçer ve
        ACTION_REGISTRY'den category/channel/priority bilgilerini alır.

        Tek Sorumluluk: hangi aksiyon detayının seçileceğine karar vermek.
        Maliyet ve uplift bilgileri optimization.py tarafından registry'den okunur.
        """
        action_detail = self._select_action_detail(row)
        spec: ActionSpec = ACTIONS[action_detail]
        note  = self._build_personalization_note(row, action_detail)

        return {
            "action_category":      spec["category"],
            "action_detail":        action_detail,
            "action_channel":       spec["channel"],
            "action_priority":      spec["priority"],
            "personalization_note": note,
        }

    def _select_action_detail(self, row: pd.Series) -> str:
        """Müşteri profiline göre aksiyon detayı string'ini seçer."""
        risk          = row.get("risk_level", "Düşük")
        monthly       = row.get("MonthlyCharges", 0)
        tenure        = row.get("tenure", 0)
        contract      = row.get("Contract", "")
        auto_payment  = row.get("UsesAutoPayment", 0)
        no_protection = row.get("NoProtectionFlag", 0)
        internet      = row.get("InternetService", "")
        num_services  = row.get("NumServices", 0)

        if risk == "Yüksek":
            if contract == "Month-to-month" and tenure <= 12 and monthly >= 80:
                return "%10 indirim + 12 ay taahhüt"
            if auto_payment == 0 and contract == "Month-to-month" and tenure > 12:
                return "otomatik ödeme teşviki + küçük indirim"
            if no_protection >= 2 and monthly < 95:
                return "koruma servis paketi + özel teklif"
            if monthly >= 95 and tenure > 24:
                return "paket gözden geçirme ve uzun dönem müşteri teklifi"
            if internet == "Fiber optic" and monthly >= 85 and contract == "Month-to-month":
                return "fiber paket sadakat indirimi + yıllık sözleşme"
            return "müşteri temsilcisi ile öncelikli temas"

        if risk == "Orta":
            if tenure <= 6:
                return "ilk dönem memnuniyet araması"
            if contract == "Month-to-month" and auto_payment == 0:
                return "otomatik ödeme teşviki"
            if no_protection >= 2:
                return "ek servis ve koruma paketi bilgilendirmesi"
            if num_services <= 2:
                return "servis genişletme teklifi ve bilgilendirme araması"
            return "memnuniyet araması ve teklif değerlendirmesi"

        return "müdahale gerekmiyor"

    @staticmethod
    def _build_personalization_note(row: pd.Series, action_detail: str) -> str:
        """Seçilen aksiyona özgü kişiselleştirme notunu oluşturur."""
        notes: dict[str, str] = {
            "%10 indirim + 12 ay taahhüt":
                "Yeni müşteri olmasına karşın yüksek fatura ödüyor ve aylık sözleşmede; "
                "indirimli teklif ile uzun vadeli bağlılık sağlanabilir.",
            "otomatik ödeme teşviki + küçük indirim":
                "Aylık sözleşmesinde otomatik ödeme bulunmuyor; her ödeme döngüsü "
                "ayrılma için fırsat yaratıyor. Küçük bir teşvik bağlılığı artırabilir.",
            "koruma servis paketi + özel teklif":
                "Güvenlik, teknik destek veya cihaz koruma paketlerinden yararlanmıyor; "
                "bu eksiklikler müşterinin platformla bağını zayıflatıyor.",
            "paket gözden geçirme ve uzun dönem müşteri teklifi":
                "Uzun süredir bizimle olan ve yüksek fatura ödeyen bu müşteriye "
                "paket optimizasyonu ile özel bir sadakat teklifi sunulabilir.",
            "fiber paket sadakat indirimi + yıllık sözleşme":
                "Fiber internet kullanıcısı yüksek aylık ödeme yapıyor ve aylık sözleşmede; "
                "yıllık taahhüt ile indirim kombinasyonu cazip bir seçenek olabilir.",
            "müşteri temsilcisi ile öncelikli temas":
                "Terk riski yüksek; belirgin bir tek risk faktörü yerine genel profil "
                "kırılganlığı söz konusu. Acil doğrudan iletişim gereklidir.",
            "ilk dönem memnuniyet araması":
                "Müşteriyle ilk 6 ay içindeyiz; bu kritik pencerede yapılan "
                "proaktif bir arama uzun vadeli bağlılığı doğrudan etkiler.",
            "otomatik ödeme teşviki":
                "Aylık sözleşmeli bu müşteri manuel ödeme yapıyor; otomatik ödemeye "
                "geçiş hem kolaylık sağlar hem de ayrılma olasılığını düşürür.",
            "ek servis ve koruma paketi bilgilendirmesi":
                "Az sayıda hizmetten yararlanan bu müşteriye uygun güvenlik veya "
                "destek paketi sunularak platforma bağlılık güçlendirilebilir.",
            "servis genişletme teklifi ve bilgilendirme araması":
                "Müşteri mevcut ürünlerimizin yalnızca küçük bir bölümünden yararlanıyor; "
                "ihtiyaçlarına göre genişletilmiş bir paket önerilebilir.",
            "memnuniyet araması ve teklif değerlendirmesi":
                "Genel memnuniyeti ölçmek ve olası endişeleri erken yakalamak için "
                "kısa bir memnuniyet görüşmesi yeterli olacaktır.",
            "müdahale gerekmiyor":
                "Müşteri sağlıklı bir ilişki profilinde görünüyor; düzenli izleme yeterlidir.",
        }
        return notes.get(action_detail, "")

    # -----------------------------------------------------------------------
    # 4. Aksiyon gerekçesi — SHAP öncelikli, kural tabanlı fallback
    # -----------------------------------------------------------------------

    def generate_reason(self, row: pd.Series) -> str:
        """
        Aksiyon gerekçesini üretir.

        Öncelik:
          1. ShapService fit edilmişse → SHAP tabanlı (model tutarlı, local XAI)
          2. ShapService yoksa        → kural tabanlı (deterministik fallback)

        Tez notu:
          SHAP tabanlı gerekçeler "post-hoc local explanation" olarak
          konumlandırılır (Lundberg & Lee, 2017). Her müşterinin bireysel
          SHAP değerlerine dayanır; gerekçe model kararıyla matematiksel
          olarak tutarlıdır.
        """
        if self._shap_svc is not None and self._feature_cols:
            return self._shap_svc.explain_row(row, self._feature_cols)
        return rule_based_reason(row)

    # -----------------------------------------------------------------------
    # 5. Dinamik uplift oranı — ACTION_REGISTRY'den
    # -----------------------------------------------------------------------

    def assign_uplift_rate(self, row: pd.Series) -> float:
        """
        Aksiyona ve müşteri özelliklerine göre tahmini elde tutma uplift'i döner.

        Temel oran: ACTION_REGISTRY'deki base_uplift değeri.
        Düzeltmeler:
          - tenure > 24 ay → +%15 (uzun süreli müşterilerde bağlılık daha güçlü)
          - tenure < 6 ay  → -%15 (henüz bağlılık oluşmamış)
          - churn_proba > 0.85 → ×0.90 (aşırı risk = düşük yanıt beklentisi)

        Tez notu:
          Bu düzeltmeler varsayımsaldır. Gerçek kampanya sonuçlarına dayalı
          lojistik regresyon veya uplift modeli ile kalibre edilmelidir.
        """
        action_detail = row.get("action", "")
        spec = get_action_spec(action_detail)

        if spec is None:
            logger.warning(
                "assign_uplift_rate: Bilinmeyen aksiyon '%s'; uplift=0.0 atandı.",
                action_detail,
            )
            return 0.0

        base       = spec["base_uplift"]
        tenure     = row.get("tenure", 0)
        churn_prob = row.get("churn_proba", 0.5)

        if tenure > 24:
            base = min(base * 1.15, 0.50)
        elif tenure < 6:
            base = base * 0.85

        if churn_prob > 0.85:
            base = base * 0.90

        return round(base, 3)

    # -----------------------------------------------------------------------
    # 6. Aksiyon ekleme
    # -----------------------------------------------------------------------

    def add_actions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tüm müşterilere aksiyon paketi, gerekçe, uplift oranı ve beklenen
        kurtarma değeri atar.
        """
        result    = df.copy()
        bundles   = result.apply(self.recommend_action_bundle, axis=1)
        bundle_df = pd.DataFrame(list(bundles))
        result    = pd.concat(
            [result.reset_index(drop=True), bundle_df.reset_index(drop=True)],
            axis=1,
        )
        result["action"]              = result["action_detail"]
        result["action_reason"]       = result.apply(self.generate_reason, axis=1)
        result["retention_uplift"]    = result.apply(self.assign_uplift_rate, axis=1)
        result["expected_saved_value"] = (
            result["expected_loss"] * result["retention_uplift"]
        )
        return result

    # -----------------------------------------------------------------------
    # 7. Ana çalıştırma
    # -----------------------------------------------------------------------

    def run(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Tam ajan pipeline'ını çalıştırır:
          iş puanları → risk seviyesi → aksiyon → öncelik sıralaması

        Döndürür
        --------
        pd.DataFrame : priority_score'a göre azalan sırada sıralanmış sonuç
        """
        logger.info("RetentionAgent başlatıldı: %d müşteri", len(predictions_df))
        result = predictions_df.copy()
        result = self.compute_business_scores(result)
        result = self.assign_risk_level(result)
        result = self.add_actions(result)
        result = result.sort_values("priority_score", ascending=False).reset_index(drop=True)
        logger.info(
            "RetentionAgent tamamlandı — Yüksek: %d, Orta: %d, Düşük: %d",
            (result["risk_level"] == "Yüksek").sum(),
            (result["risk_level"] == "Orta").sum(),
            (result["risk_level"] == "Düşük").sum(),
        )
        return result
