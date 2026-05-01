"""
llm_service.py
Müşteri tutundurma platformu — yorum servisi.

Mimari karar (tez revizyonu):
- Portföy özeti ve strateji karşılaştırması → tamamen kural tabanlı (deterministik)
- Tekil müşteri yorumu → LLM (Qwen2.5-7B, llama.cpp)

Gerekçe:
  Portföy ve strateji metinleri sayısal eşik kararlarına dayanır; bu kararlar
  kural tabanlı üretildiğinde hem tekrarlanabilir (reproducible) hem de
  akademik açıdan savunulabilir. "Açıklanabilir AI" katmanı olarak tezde
  açıkça konumlandırılmıştır.
  LLM yalnızca müşteri temsilcisine yönelik, bağlamsal ve doğal dil gerektiren
  kişiselleştirilmiş yorumlarda kullanılır.

LLM düzeltmeleri (saçmalama giderme):
  - temperature: 0.1  (determinizm)
  - max_tokens: 120   (kısa tut, model dağılmasın)
  - repeat_penalty: 1.1
  - Prompt: format instruction kaldırıldı, sadece görev tanımı bırakıldı
  - Çıktı temizleme: Markdown, madde işareti, fazla cümle
"""

import re
import logging
import requests
from typing import Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

_BASE_URL       = "http://localhost:8080/v1/chat/completions"
_MODEL_NAME     = "qwen2.5-7b"
_TIMEOUT        = 60
_DEFAULT_TOKENS = 120

_llm_unavailable_warned = False  # her process başına bir kez uyar


# ---------------------------------------------------------------------------
# Yardımcı: Çıktı temizleme
# ---------------------------------------------------------------------------
def _clean(text: str, max_sentences: int = 2) -> str:
    if not text:
        return text
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'^[-•]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r' {2,}', ' ', text).strip()
    # "Cümle 1 —", "1." gibi yapay etiketleri sil
    text = re.sub(r'(Cümle\s*\d+\s*[—\-:.]?\s*)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\d+[.)]\s*', '', text)
    # Cümle sayısını kısıtla
    parts = re.split(r'(?<=[.!?])\s+', text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > max_sentences:
        text = ' '.join(parts[:max_sentences])
        if not text[-1] in '.!?':
            text += '.'
    return text


# ---------------------------------------------------------------------------
# Yardımcı: API çağrısı
# ---------------------------------------------------------------------------
def _call_llm(
    system: str,
    user: str,
    max_tokens: int = _DEFAULT_TOKENS,
    max_sentences: int = 2,
) -> Optional[str]:
    payload = {
        "model":          _MODEL_NAME,
        "messages":       [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "max_tokens":     max_tokens,
        "temperature":    0.1,
        "top_p":          0.9,
        "repeat_penalty": 1.1,
        "stream":         False,
    }
    global _llm_unavailable_warned
    try:
        r = requests.post(
            _BASE_URL, json=payload, timeout=_TIMEOUT,
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()
        _llm_unavailable_warned = False  # başarılı bağlantıda sıfırla
        raw = r.json()["choices"][0]["message"]["content"].strip()
        return _clean(raw, max_sentences=max_sentences)
    except requests.exceptions.ConnectionError:
        if not _llm_unavailable_warned:
            logger.warning(
                "llama.cpp server'a bağlanılamadı — kural tabanlı yorum kullanılıyor. "
                "(Yerel Qwen 2.5 sunucusu çalışmıyor; bu normal bir durumdur.)"
            )
            _llm_unavailable_warned = True
        return None
    except requests.exceptions.Timeout:
        logger.warning("llama.cpp zaman aşımı (>%ds).", _TIMEOUT)
        return None
    except Exception as exc:
        logger.warning("LLM hatası: %s", exc)
        return None


# ===========================================================================
# Kural tabanlı metin üretimi  (portföy + strateji)
# ===========================================================================

def _risk_label(avg_churn: float, high_ratio: float) -> str:
    if avg_churn >= 0.65 or high_ratio >= 0.60:
        return "kritik düzeyde yüksek"
    if avg_churn >= 0.40 or high_ratio >= 0.30:
        return "orta-yüksek"
    return "görece dengeli"


def rule_based_portfolio_summary(summary: dict) -> str:
    """
    Portföy değerlendirmesini deterministik kural seti ile üretir.
    Yönetici ve müşteri temsilcisine yönelik, teknik terimlerden arındırılmış dil.
    LLM kullanılmaz — tekrarlanabilir ve savunulabilir çıktı.
    """
    total     = summary.get("total_customers", 0)
    high      = summary.get("high_risk_count", 0)
    mid       = summary.get("medium_risk_count", 0)
    avg_churn = summary.get("avg_churn", 0.0)
    loss      = summary.get("portfolio_expected_loss", 0.0)
    urgency   = summary.get("urgency", "")
    saved     = summary.get("selected_expected_saved", 0.0)
    roi       = summary.get("avg_roi", 0.0)
    sel       = summary.get("selected_count", 0)
    budget    = summary.get("selected_budget", 0.0)

    high_ratio = high / max(total, 1)
    risk_label = _risk_label(avg_churn, high_ratio)

    # Portföy durum cümlesi — teknik terimlerden arındırılmış
    durum = (
        f"Toplam {total:,} müşteriden {high:,} tanesi yüksek, "
        f"{mid:,} tanesi orta terk riski taşımaktadır "
        f"(%{high_ratio*100:.0f} kritik segment). "
        f"Portföydeki risk düzeyi <b>{risk_label}</b> olarak değerlendirilmekte; "
        f"müşteri kaybına bağlı tahmini gelir etkisi <b>{loss:,.0f} TL</b>'dir."
    )

    # Öneri cümlesi — karar vericiye yönelik net çağrı
    if urgency.startswith("Acil"):
        aksiyon = (
            f"Öncelikle aksiyon alınması gereken <b>{sel:,} müşteri</b> belirlenmiştir. "
            f"Toplam <b>{budget:,.0f} TL</b> kampanya yatırımıyla "
            f"<b>{saved:,.0f} TL</b> gelirin korunması hedeflenmektedir "
            f"(ortalama geri dönüş: <b>{roi:.1f}x</b>). "
            f"Bu müşteriler için derhal iletişim başlatılması önerilir."
        )
    elif urgency.startswith("Öncelikli"):
        aksiyon = (
            f"<b>{sel:,} müşteri</b> öncelikli aksiyon listesine alınmıştır. "
            f"Kampanya yatırımı <b>{budget:,.0f} TL</b>, "
            f"kurtarılması beklenen gelir <b>{saved:,.0f} TL</b> "
            f"(ortalama geri dönüş: <b>{roi:.1f}x</b>)."
        )
    else:
        aksiyon = (
            f"Portföy genel itibarıyla dengeli bir görünüm sergilemektedir. "
            f"<b>{sel:,} müşteri</b> proaktif izleme kapsamına alınmış olup "
            f"kurtarılması hedeflenen gelir <b>{saved:,.0f} TL</b>'dir."
        )

    return f"{durum}<br><br>{aksiyon}"


def rule_based_strategy_comment(comparison_df: pd.DataFrame, advantage: dict) -> str:
    """
    Strateji karşılaştırma yorumunu deterministik kural seti ile üretir.
    """
    agent_is_best  = advantage.get("agent_is_best", False)
    vs_bl          = advantage.get("vs_baseline_pct", 0.0)
    vs_ro          = advantage.get("vs_risk_only_pct", 0.0)

    # Agent satırını bul
    agent_row = comparison_df[
        comparison_df["strategy"].str.contains("Yapay Zekâ", na=False)
    ]
    if len(agent_row) == 0:
        return "Strateji karşılaştırma verisi yetersiz."

    agent_nb  = float(agent_row["net_benefit"].values[0])
    agent_roi = float(agent_row["avg_roi"].values[0])

    if agent_is_best:
        return (
            f"Yapay Zekâ Destekli Akıllı Yaklaşım, Geleneksel Toplu Yaklaşım'a kıyasla "
            f"%{vs_bl:.1f} daha fazla, Risk Bazlı Sabit Aksiyon'a kıyasla %{vs_ro:.1f} daha fazla "
            f"net kazanç ({agent_nb:,.0f} TL, ortalama geri dönüş {agent_roi:.2f}x) sağlamaktadır. "
            f"Her müşteriye özel aksiyon atanması ve bütçe kısıtı altında öncelik sıralı seçim "
            f"bu üstünlüğün temel kaynağıdır."
        )
    else:
        return (
            f"Mevcut bütçe veya aday havuzu büyüklüğü kısıtları nedeniyle Yapay Zekâ Destekli "
            f"Akıllı Yaklaşım en yüksek net kazancı sağlayamamıştır ({agent_nb:,.0f} TL, "
            f"geri dönüş {agent_roi:.2f}x). Kampanya bütçesinin artırılması veya "
            f"daha geniş bir müşteri havuzunun değerlendirmeye alınması önerilir."
        )


# ===========================================================================
# LLMService
# ===========================================================================
# ---------------------------------------------------------------------------
# Kural tabanlı tekil müşteri yorumu — LLM kullanılamadığında devreye girer
# ---------------------------------------------------------------------------
def _rule_based_customer_comment(row: pd.Series) -> str:
    """
    LLM bağlantısı olmadığında deterministik, iş dostu Türkçe yorum üretir.
    Ham alan adları (InternetService vb.) kullanılmaz.
    """
    monthly   = float(row.get("MonthlyCharges", 0) or 0)
    contract  = str(row.get("Contract", "") or "")
    churn_pct = float(row.get("churn_proba", 0) or 0) * 100
    action    = str(row.get("action_detail", "—") or "—")
    roi       = float(row.get("roi", 0) or 0)
    tenure    = int(row.get("tenure", 0) or 0)

    # Sözleşme bağlamı
    if "Month-to-month" in contract or "month" in contract.lower():
        bag = "aylık sözleşmede olduğu için ayrılma maliyeti çok düşük"
    elif "One year" in contract or "one" in contract.lower():
        bag = "yıllık sözleşmesi var; yenileme döneminde müdahale kritik"
    elif "Two year" in contract or "two" in contract.lower():
        bag = "2 yıllık sözleşmesi olmasına rağmen risk eşiğinin üzerinde"
    else:
        bag = "mevcut sözleşme koşulları risk oluşturuyor"

    # Müşteri deneyim süresi
    if tenure <= 6:
        deneyim = "henüz yeni bir müşteri"
    elif tenure <= 24:
        deneyim = f"{tenure} aydır müşterimiz"
    else:
        deneyim = f"{tenure} ay gibi uzun süredir müşterimiz"

    # Risk yoğunluğu
    if churn_pct >= 80:
        risk_ifade = "son derece yüksek"
    elif churn_pct >= 60:
        risk_ifade = "belirgin biçimde yüksek"
    else:
        risk_ifade = "dikkat gerektiren"

    return (
        f"Bu müşteri {deneyim}; aylık {monthly:.0f} TL ödüyor ve {bag}. "
        f"Terk riski %{churn_pct:.0f} ile {risk_ifade} düzeyde olduğundan "
        f"'{action}' aksiyonu önerilmektedir "
        f"({roi:.1f}x yatırım geri dönüşü beklenmektedir)."
    )


class LLMService:
    """
    Müşteri tutundurma yorumlama servisi.

    Portföy ve strateji özetleri → kural tabanlı (deterministik, savunulabilir).
    Tekil müşteri yorumu       → LLM (Qwen2.5-7B, bağlamsal doğal dil).
    """

    # -----------------------------------------------------------------------
    # 1. Portföy özeti — kural tabanlı
    # -----------------------------------------------------------------------
    def generate_portfolio_summary(self, summary: dict) -> str:
        return rule_based_portfolio_summary(summary)

    # -----------------------------------------------------------------------
    # 2. Tekil müşteri yorumu — LLM + Guardrail
    # -----------------------------------------------------------------------
    def generate_customer_comment(self, row: pd.Series) -> Optional[str]:
        system = (
            "Sen bir telekom şirketinin müşteri tutundurma analistisın. "
            "Verilen müşteri profilini inceleyip müşteri temsilcisine "
            "kısa, somut, Türkçe bir yorum yaz. "
            "İki cümleyi geçme. Madde işareti veya başlık kullanma."
        )

        churn_pct = row.get("churn_proba", 0) * 100
        monthly   = row.get("MonthlyCharges", 0)
        contract  = row.get("Contract", "—")
        action    = row.get("action_detail", "—")
        roi       = row.get("roi", 0)

        # En belirleyici tek risk faktörünü seç (iş dili)
        if row.get("UsesAutoPayment", 1) == 0 and contract == "Month-to-month":
            faktor = "otomatik ödeme kullanmıyor, aylık sözleşmede — ayrılması çok kolay"
        elif row.get("NoProtectionFlag", 0) >= 2:
            faktor = "güvenlik ve destek hizmetlerine sahip değil"
        elif contract == "Month-to-month" and monthly >= 80:
            faktor = "yüksek fatura ödüyor ancak uzun vadeli bir bağı yok"
        elif row.get("tenure", 999) <= 12:
            faktor = "yeni müşteri, bağlılık henüz oluşmamış"
        else:
            faktor = "birden fazla risk faktörü bir arada"

        user = (
            f"Müşteri profili: {monthly:.0f} TL aylık fatura, "
            f"sözleşme türü {contract}, terk olasılığı %{churn_pct:.1f}. "
            f"Dikkat çeken durum: {faktor}. "
            f"Önerilen aksiyon: {action} (yatırım geri dönüşü {roi:.1f}x). "
            f"Bu müşteriye neden bu teklifi yapmalıyız?"
        )

        raw_comment = _call_llm(system, user, max_tokens=120, max_sentences=2)

        # LLM ulaşılamıyor veya boş döndü → kural tabanlı fallback
        if not raw_comment:
            return _rule_based_customer_comment(row)

        # Guardrail doğrulaması — ham LLM çıktısını filtrele
        guardrail = LLMGuardrail()
        context   = row.to_dict()
        _, _, safe_comment = guardrail.validate(
            raw_comment, context,
            lambda ctx: _rule_based_customer_comment(pd.Series(ctx)),
        )
        return safe_comment

    # -----------------------------------------------------------------------
    # 3. DataFrame'e toplu yorum ekleme
    # -----------------------------------------------------------------------
    def add_llm_comment(
        self,
        df: pd.DataFrame,
        limit: int = 6,
        comment_col: str = "llm_comment",
    ) -> pd.DataFrame:
        result = df.copy()
        result[comment_col] = None
        for i, (idx, row) in enumerate(result.iterrows()):
            if i >= limit:
                break
            result.at[idx, comment_col] = self.generate_customer_comment(row)
        return result

    # -----------------------------------------------------------------------
    # 4. Strateji karşılaştırması — kural tabanlı
    # -----------------------------------------------------------------------
    def generate_strategy_comparison_comment(
        self,
        comparison_df: pd.DataFrame,
        advantage: dict,
    ) -> str:
        return rule_based_strategy_comment(comparison_df, advantage)


# ===========================================================================
# LLM Guardrail — Güvenlik ve Doğrulama Katmanı
#
# Akademik motivasyon:
#   LLM (Qwen 2.5) çıktıları deterministik değildir ve bazen sayısal
#   tutarsızlık, uzunluk aşımı veya uygunsuz ifade içerebilir. Bu katman
#   üretilen metni üç eksende denetler ve hata durumunda deterministik
#   kural tabanlı fallback'i devreye sokar.
#
#   Denetim eksenleri:
#     1. Küfür / argo filtresi    — Türkçe uygunsuz kelime listesi
#     2. Sayısal tutarsızlık      — LLM'in ürettiği rakam,
#                                    bağlam değerleriyle çelişiyor mu?
#     3. Maksimum uzunluk         — Token sayısı güvenli aralıkta mı?
#
#   Mimari karar:
#     Guardrail, LLMService içine gömülmek yerine ayrı bir sınıf olarak
#     tanımlanmıştır; bu sayede test edilebilirlik ve genişletilebilirlik
#     sağlanır (Open/Closed Principle).
# ===========================================================================

class LLMGuardrail:
    """
    LLM yanıtlarını doğrulayan ve güvenli fallback sağlayan kalkan katmanı.

    Kullanım:
        guardrail = LLMGuardrail()
        ok, reason, safe_text = guardrail.validate(llm_output, context, fallback_fn)
        if not ok:
            use(safe_text)  # fallback
    """

    # Türkçe argo / uygunsuz ifade listesi (kısa ama temsili)
    _PROFANITY: frozenset = frozenset({
        "salak", "aptal", "ahmak", "gerizekalı", "mal", "dangalak",
        "s*ktir", "orospu", "piç", "göt", "oç", "amk", "bok",
        "lanet", "kahpe", "sürtük", "ibne",
    })

    MAX_CHARS      = 600    # Tek yorum için maksimum karakter sayısı
    NUMBER_PATTERN = re.compile(r"\b\d[\d.,]*\b")

    # -----------------------------------------------------------------------
    # Ana doğrulama metodu
    # -----------------------------------------------------------------------

    def validate(
        self,
        response:    str,
        context:     dict,
        fallback_fn: "callable",
    ) -> Tuple[bool, str, str]:
        """
        LLM yanıtını üç filtreden geçirir.

        Parametreler
        ------------
        response    : LLM'in ürettiği ham metin
        context     : Bağlam sözlüğü (row verileri: churn_proba, roi, vb.)
        fallback_fn : Guardrail başarısız olursa çağrılacak kural tabanlı fonksiyon

        Döndürür
        --------
        (geçti: bool, red_nedeni: str, kullanılacak_metin: str)
        """
        if not response or not response.strip():
            return False, "Boş yanıt", fallback_fn(context)

        # 1 — Küfür / argo filtresi
        ok, reason = self._check_profanity(response)
        if not ok:
            logger.warning("Guardrail [küfür]: %s", reason)
            return False, reason, fallback_fn(context)

        # 2 — Sayısal tutarsızlık filtresi
        ok, reason = self._check_numerical_consistency(response, context)
        if not ok:
            logger.warning("Guardrail [sayısal tutarsızlık]: %s", reason)
            return False, reason, fallback_fn(context)

        # 3 — Uzunluk filtresi
        ok, reason = self._check_max_length(response)
        if not ok:
            logger.warning("Guardrail [uzunluk]: %s", reason)
            # Uzunluk aşımında kırp; tamamen reddetme
            clipped = response[: self.MAX_CHARS].rsplit(".", 1)[0] + "."
            return False, reason, clipped

        return True, "ok", response

    # -----------------------------------------------------------------------
    # Filtre 1: Küfür / argo
    # -----------------------------------------------------------------------

    def _check_profanity(self, text: str) -> Tuple[bool, str]:
        lower = text.lower()
        for word in self._PROFANITY:
            if word in lower:
                return False, f"Uygunsuz kelime tespit edildi: '{word}'"
        return True, "ok"

    # -----------------------------------------------------------------------
    # Filtre 2: Sayısal tutarsızlık
    # -----------------------------------------------------------------------

    def _check_numerical_consistency(
        self, text: str, context: dict
    ) -> Tuple[bool, str]:
        """
        LLM'in ürettiği sayısal değerlerin bağlamdaki gerçek değerlerle
        makul ölçüde örtüşüp örtüşmediğini kontrol eder.

        Kural: LLM metindeki her sayı, bağlamdaki en yakın sayısal değerden
        %200'den fazla sapıyorsa tutarsız kabul edilir.

        Bu eşik kasıtlı olarak gevşek tutulmuştur; LLM doğal dilde
        yuvarlama veya birim dönüşümü yapabilir.
        """
        nums_in_text = [
            float(n.replace(",", "."))
            for n in self.NUMBER_PATTERN.findall(text)
            if self._is_numeric_token(n)
        ]
        if not nums_in_text:
            return True, "ok"

        # Bağlamdan karşılaştırılabilir sayısal değerler
        context_nums = []
        for key in ("churn_proba", "roi", "MonthlyCharges", "estimated_clv",
                    "net_benefit", "expected_loss", "offer_cost"):
            val = context.get(key)
            if val is not None:
                try:
                    context_nums.append(float(val))
                except (TypeError, ValueError):
                    pass

        if not context_nums:
            return True, "ok"  # Karşılaştırma yapılamıyorsa geç

        for num in nums_in_text:
            if num == 0:
                continue
            # Bağlamdaki en yakın değere olan oransal fark
            min_ratio = min(
                abs(num - c) / max(abs(c), 1e-6) for c in context_nums
            )
            if min_ratio > 2.0:  # %200'den fazla sapma → şüpheli
                return (
                    False,
                    f"Sayısal tutarsızlık: LLM {num:.1f} üretti, "
                    f"bağlamda en yakın değer {min(context_nums, key=lambda c: abs(num - c)):.1f}",
                )
        return True, "ok"

    @staticmethod
    def _is_numeric_token(token: str) -> bool:
        """Noktalama içeren token'ın gerçek sayı olup olmadığını test eder."""
        try:
            float(token.replace(",", "."))
            return True
        except ValueError:
            return False

    # -----------------------------------------------------------------------
    # Filtre 3: Uzunluk denetimi
    # -----------------------------------------------------------------------

    def _check_max_length(self, text: str) -> Tuple[bool, str]:
        if len(text) > self.MAX_CHARS:
            return (
                False,
                f"Yanıt çok uzun ({len(text)} karakter > {self.MAX_CHARS} sınır); kırpıldı.",
            )
        return True, "ok"