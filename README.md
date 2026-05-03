# 📊  Müşteri Tutundurma Zekâsı Platformu (Retention Intelligence Platform)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg) ![Flask](https://img.shields.io/badge/Flask-3.0-green.svg) ![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

Bu proje, müşteri kaybını (**churn**) önceden tahmin etmek ve müşteri geri bildirimlerini analiz ederek işletmelere stratejik kararlar aldırmak amacıyla geliştirilmiş uçtan uca bir **Veri Bilimi ve Web Platformudur**.

---

## 🚀 Proje Hakkında
İşletmeler için yeni müşteri kazanmak, mevcut müşteriyi tutmaktan çok daha maliyetlidir. Bu platform şu temel işlevleri sunar:

* **Tahminleme:** Makine öğrenmesi modelleri ile hangi müşterilerin ayrılma eğiliminde olduğunu belirler.
* **Analiz:** Çok faktörlü yorum motoru ile müşteri geri bildirimlerini anlamlandırır.
* **Hibrit Zekâ:** Kaynak yönetimi için hem yerel LLM (Llama) hem de bulut tabanlı kural motoru kullanır.

---

## 🏗️ Teknik Mimari
Proje, modüler ve sürdürülebilir bir yapı üzerine inşa edilmiştir:

* **Backend:** Flask (Daha yüksek özelleştirme ve kontrol için Streamlit'ten geçiş yapılmıştır).
* **Makine Öğrenmesi:** Veri işleme ve modelleme süreçleri ağırlıklı olarak **Python (%79.2)** ile gerçekleştirilmiştir.
* **LLM Entegrasyonu:** Müşteri yorumlarını derinlemesine analiz etmek için **Llama Server** entegrasyonu mevcuttur.
* **Deployment:** Uygulama; Render veya Railway gibi bulut ortamlarında stabil çalışması için kural tabanlı bir yedekleme (**fallback**) mekanizmasına sahiptir.

---

## 📂 Dosya Yapısı
Proje klasörleme mantığı profesyonel standartlara uygundur:

* **`artifacts/`** : Eğitilmiş model dosyaları (.pkl, .h5 vb.).
* **`data/`** : Ham ve işlenmiş veri setleri.
* **`src/`** : Uygulamanın çekirdek fonksiyonları ve yardımcı scriptler.
* **`train/`** : Model eğitim süreçlerine ait kodlar.
* **`static/`** & **`templates/`** : Web arayüzü (UI) dosyaları.
* **`flask_app.py`** : Uygulamanın ana giriş noktası.

---

## 🛠️ Kurulum ve Çalıştırma

### 1. Yerel Kurulum (Full Model + LLM)
Yerel ortamda tüm özellikleri (LLM dahil) kullanmak için:
```bash
git clone [https://github.com/GulsumSayin/retention-system.git](https://github.com/GulsumSayin/retention-system.git)
cd retention-system
pip install -r requirements.txt
python flask_app.py
```
Not: Yerelde AI yorumlarını görmek için ilgili LLM model dosyasının (.gguf) yerelde mevcut olması gerekir.

### 2. Bulut Dağıtımı (Render / Railway)

Proje; **Procfile**, **runtime.txt** ve **railway.json** dosyalarıyla bulut ortamları için optimize edilmiştir. Bulut sürümünde sistem, kaynak tasarrufu için otomatik olarak **Kural Tabanlı Yorum Motoruna** geçiş yapar.

---

## 🧠 Hibrit Yorum Motoru

Proje, donanım kısıtlamalarına uyum sağlamak için iki farklı modda çalışabilir:

* **AI Destekli Mod (Local):** Yüksek işlem gücü ile derinlemesine duygu ve içerik analizi yapar.
* **Çevik Mod (Web):** Kural tabanlı mantık ile düşük kaynak tüketerek hızlı sonuç üretir[cite: 69, 76, 84.
