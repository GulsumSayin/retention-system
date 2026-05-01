"""
action_registry.py
Merkezi aksiyon kataloğu — tek gerçek kaynak (Single Source of Truth).

Tasarım gerekçesi:
  agents.py ve optimization.py, daha önce aynı aksiyon adı stringlerini
  bağımsız dict'lerde tanımlıyordu; string uyuşmazlığında optimization.py
  sessizce 0 maliyet hesaplayıp müşteriyi filtreden düşürüyordu.
  Bu modül her ikisi için de tek referans noktasıdır.

Aksiyon yapısı (ActionSpec):
  category     : İş sürecindeki üst kategori
  channel      : Uygulama kanalı
  priority     : Aksiyon önceliği
  base_uplift  : Temel elde tutma artış oranı (0–1)
  cost_type    : "rate"  → aylık ücretin yüzdesi
                 "fixed" → sabit TL tutarı
  cost_value   : cost_type="rate"  ise çarpan (ör. 0.20 = %20)
                 cost_type="fixed" ise TL (ör. 25.0)
  cost_note    : Maliyet varsayımının açıklaması (akademik şeffaflık)

Tez notu:
  Tüm uplift ve maliyet değerleri iş bilgisine dayalı varsayımsaldır.
  Gerçek kampanya verisiyle kalibre edilmeleri gerekir (A/B testi veya
  Difference-in-Differences yöntemi önerilir).
"""

from typing import TypedDict


class ActionSpec(TypedDict):
    category:    str
    channel:     str
    priority:    str
    base_uplift: float
    cost_type:   str    # "rate" | "fixed"
    cost_value:  float
    cost_note:   str


# ---------------------------------------------------------------------------
# Aksiyon detayı (action_detail string) → ActionSpec
# ---------------------------------------------------------------------------
ACTIONS: dict[str, ActionSpec] = {

    # ── Yüksek risk — Fiyat Müdahalesi ────────────────────────────────────
    "%10 indirim + 12 ay taahhüt": ActionSpec(
        category="Fiyat Müdahalesi",
        channel="Çağrı Merkezi",
        priority="Yüksek",
        base_uplift=0.35,
        cost_type="rate",
        cost_value=0.20,
        cost_note=(
            "Kampanya toplam maliyeti: %10 indirim NPV (yaklaşık 1.2 aylık ücret, "
            "12 ay vade) + çağrı merkezi operasyon bedeli. "
            "Toplam ≈ %20 aylık ücret olarak modellenmiştir."
        ),
    ),

    # ── Yüksek risk — Ödeme Davranışı ─────────────────────────────────────
    "otomatik ödeme teşviki + küçük indirim": ActionSpec(
        category="Ödeme Davranışı Müdahalesi",
        channel="SMS + Çağrı Merkezi",
        priority="Yüksek",
        base_uplift=0.20,
        cost_type="rate",
        cost_value=0.10,
        cost_note="Teşvik indirimi (%5–8) + SMS/çağrı operasyon bedeli ≈ %10 aylık ücret.",
    ),

    # ── Yüksek risk — Hizmet Güçlendirme ──────────────────────────────────
    "koruma servis paketi + özel teklif": ActionSpec(
        category="Hizmet Güçlendirme",
        channel="Çağrı Merkezi",
        priority="Yüksek",
        base_uplift=0.25,
        cost_type="rate",
        cost_value=0.15,
        cost_note=(
            "İlk ay ücretsiz servis paketi teklifi (~%10 aylık ücret) "
            "+ çağrı merkezi operasyon bedeli (~%5) ≈ %15 aylık ücret."
        ),
    ),

    # ── Yüksek risk — Paket Revizyonu ─────────────────────────────────────
    "paket gözden geçirme ve uzun dönem müşteri teklifi": ActionSpec(
        category="Paket Revizyonu",
        channel="Çağrı Merkezi",
        priority="Yüksek",
        base_uplift=0.18,
        cost_type="fixed",
        cost_value=20.0,
        cost_note="Uzman danışmanlık görüşmesi (20 TL sabit operasyon bedeli).",
    ),

    # ── Yüksek risk — Fiber Sadakat ───────────────────────────────────────
    "fiber paket sadakat indirimi + yıllık sözleşme": ActionSpec(
        category="Fiyat Müdahalesi",
        channel="Çağrı Merkezi",
        priority="Yüksek",
        base_uplift=0.30,
        cost_type="rate",
        cost_value=0.18,
        cost_note=(
            "Fiber sadakat indirimi (~%12 aylık ücret, 12 ay taahhüt NPV) "
            "+ çağrı merkezi operasyon bedeli (~%6) ≈ %18 aylık ücret."
        ),
    ),

    # ── Yüksek risk — Genel Temas ─────────────────────────────────────────
    "müşteri temsilcisi ile öncelikli temas": ActionSpec(
        category="Öncelikli Temas",
        channel="Çağrı Merkezi",
        priority="Yüksek",
        base_uplift=0.15,
        cost_type="fixed",
        cost_value=25.0,
        cost_note="Öncelikli inbound/outbound çağrı maliyeti (25 TL sabit).",
    ),

    # ── Orta risk — Erken Dönem ───────────────────────────────────────────
    "ilk dönem memnuniyet araması": ActionSpec(
        category="Erken Dönem Tutundurma",
        channel="Çağrı Merkezi",
        priority="Orta",
        base_uplift=0.10,
        cost_type="fixed",
        cost_value=15.0,
        cost_note="Standart memnuniyet araması (15 TL operasyon bedeli).",
    ),

    # ── Orta risk — Ödeme Teşviki ─────────────────────────────────────────
    "otomatik ödeme teşviki": ActionSpec(
        category="Ödeme Davranışı Müdahalesi",
        channel="SMS + Çağrı Merkezi",
        priority="Orta",
        base_uplift=0.08,
        cost_type="fixed",
        cost_value=12.0,
        cost_note="SMS otomasyonu + kısa takip araması (12 TL).",
    ),

    # ── Orta risk — Hizmet Bilgilendirme ──────────────────────────────────
    "ek servis ve koruma paketi bilgilendirmesi": ActionSpec(
        category="Hizmet Güçlendirme",
        channel="Çağrı Merkezi",
        priority="Orta",
        base_uplift=0.09,
        cost_type="fixed",
        cost_value=10.0,
        cost_note="Bilgilendirme araması (10 TL operasyon bedeli).",
    ),

    # ── Orta risk — Çapraz Satış ──────────────────────────────────────────
    "servis genişletme teklifi ve bilgilendirme araması": ActionSpec(
        category="Çapraz Satış",
        channel="SMS + Çağrı Merkezi",
        priority="Orta",
        base_uplift=0.10,
        cost_type="fixed",
        cost_value=10.0,
        cost_note="SMS + takip araması (10 TL operasyon bedeli).",
    ),

    # ── Orta risk — Memnuniyet ────────────────────────────────────────────
    "memnuniyet araması ve teklif değerlendirmesi": ActionSpec(
        category="Memnuniyet Teması",
        channel="Çağrı Merkezi",
        priority="Orta",
        base_uplift=0.08,
        cost_type="fixed",
        cost_value=10.0,
        cost_note="Rutin memnuniyet araması (10 TL operasyon bedeli).",
    ),

    # ── Düşük risk — İzleme ───────────────────────────────────────────────
    "müdahale gerekmiyor": ActionSpec(
        category="İzleme",
        channel="Yok",
        priority="Düşük",
        base_uplift=0.00,
        cost_type="fixed",
        cost_value=0.0,
        cost_note="Aktif kampanya yok; periyodik sistem izlemesi.",
    ),
}


def get_action_spec(action_detail: str) -> ActionSpec | None:
    """
    Verilen aksiyon detayına karşılık gelen ActionSpec'i döner.
    Tanımsız aksiyon için None döner (çağıran taraf uyarı loglamalıdır).
    """
    return ACTIONS.get(action_detail)


def all_action_details() -> list[str]:
    """Kayıtlı tüm aksiyon detay stringlerini döner."""
    return list(ACTIONS.keys())
