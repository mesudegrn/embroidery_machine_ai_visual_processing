import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math

# ======== SAYFA & STİL ========
st.set_page_config(page_title="KIZILAY BARINMA SİSTEMLERİ", page_icon="🧵", layout="centered")

st.markdown(
    """
    <style>
      .logo-bg {
        position: fixed;
        top: 25px; right: 25px;
        width: 260px; opacity: 0.15;
        z-index: -1; pointer-events: none;
      }
      .kizilay-baslik {
        font-size: 50px; color: #d50000; font-weight: 800;
        text-align: center; margin: 10px 0 0 0;
      }
      .kizilay-subtitle {
        font-size: 24px; color: #555; text-align: center; margin: 6px 0 24px 0;
      }
    </style>
    <img class="logo-bg" src="kizilay_logo.png">
    <div class="kizilay-baslik">KIZILAY BARINMA SİSTEMLERİ</div>
    <div class="kizilay-subtitle">🧵 Nakış Makinesi Üretim Süresi ve Verimlilik Tahmini</div>
    """,
    unsafe_allow_html=True
)

# ======== MODELİ YÜKLE ========
prod_model = None
try:
    prod_model = joblib.load("model_ascii.pkl")
except Exception:
    st.error("Üretim modeli bulunamadı. Aynı klasöre 'model_ascii.pkl' koy.")
    st.stop()


# ======== GİRİŞLER ========
col1, col2 = st.columns(2)
with col1:
    fabric_count = st.number_input("Kumaş Sayısı", min_value=1, value=10, step=1)
    num_workers  = st.number_input("İşçi Sayısı", min_value=1, value=3, step=1)
with col2:
    machine_speed = st.number_input("Makine Hızı (dk)", min_value=1, max_value=6, value=3, step=1)
    embroidery_type = st.selectbox("Nakış Türü", ["gri_arma", "kirmizi_arma", "SAR_arma"], index=0)

# ======== YARDIMCI: ÖZNITELIK EŞLEŞTİRME ========
def get_expected_features(model):
    # Sklearn 1.0+ için:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    # Pipeline içinde olabilir
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    # Bilemediysek en yaygın Türkçe seti dön
    return ["kumaş_sayısı", "makine_hızı_(dk)", "i̇şçi_sayısı", "nakış_türü_gri_arma"]

def build_feature_row(fabric_count, machine_speed, num_workers, embroidery_type, expected_cols):
    # Kullanıcı girişlerinden olası tüm alias’ları üret (noktalı i, vs.)
    base_aliases = {
        "fabric_count": ["fabric_count", "kumaş_sayısı", "kumas_sayisi"],
        "machine_time_min": ["machine_time_min", "makine_hızı_(dk)", "makine_hizi_dk"],
        "num_of_workers": ["num_of_workers", "i̇şçi_sayısı", "işçi_sayısı", "isci_sayisi"],
        # one-hot’lar
        "embroidery_type_gri_arma": ["embroidery_type_gri_arma", "nakış_türü_gri_arma", "nakis_turu_gri_arma"],
        "embroidery_type_kirmizi_arma": ["embroidery_type_kirmizi_arma", "nakış_türü_kirmizi_arma", "nakis_turu_kirmizi_arma"],
        "embroidery_type_sar_arma": ["embroidery_type_sar_arma", "nakış_türü_sar_arma", "nakis_turu_sar_arma"],
    }

    # Kullanıcı girişlerinden canonical değerler
    canonical = {
        "fabric_count": fabric_count,
        "machine_time_min": machine_speed,
        "num_of_workers": num_workers,
        "embroidery_type_gri_arma": 1 if embroidery_type.lower()=="gri_arma" else 0,
        "embroidery_type_kirmizi_arma": 1 if embroidery_type.lower()=="kirmizi_arma" else 0,
        "embroidery_type_sar_arma": 1 if embroidery_type.lower()=="sar_arma" else 0,
    }

    row = {}
    for col in expected_cols:
        # expected col hangi canonical alanın alias’ı ise onu doldur
        filled = False
        for key, aliases in base_aliases.items():
            if col in aliases or col == key:
                row[col] = canonical[key]
                filled = True
                break
        if not filled:
            # Temel 3 sayısal kolonlardan birine doğrudan oturabilir
            if col in ["fabric_count", "kumaş_sayısı", "kumas_sayisi"]:
                row[col] = fabric_count
            elif col in ["machine_time_min", "makine_hızı_(dk)", "makine_hizi_dk"]:
                row[col] = machine_speed
            elif col in ["num_of_workers", "i̇şçi_sayısı", "işçi_sayısı", "isci_sayisi"]:
                row[col] = num_workers
            else:
                # Bilinmeyen her şeyi 0 doldur (one-hot/boş kolonlar)
                row[col] = 0
    return pd.DataFrame([row])

expected_cols = get_expected_features(prod_model)

# ======== TAHMİN ET BUTONU ========
# ======== TAHMİN ET BUTONU ========
if st.button("🔍 Tahmin Et"):
    # Girdi DataFrame’i, modelin beklediği kolon adlarına göre kur
    X_input = build_feature_row(fabric_count, machine_speed, num_workers, embroidery_type, expected_cols)

    try:
        predicted_total_time = float(prod_model.predict(X_input)[0])  # saniye varsayımı
    except Exception as e:
        st.error(f"Model tahmininde hata: {e}")
        st.stop()

    unit_time = predicted_total_time / fabric_count  # saniye / adet
    max_daily_production = max(0, math.floor((450 * 60) / unit_time))

    st.subheader("⏱️ Tahmin Sonuçları")
    st.write(f"**Tahmini Toplam Süre:** {round(predicted_total_time, 2)} saniye")
    st.write(f"**Birim Süre:** {round(unit_time, 2)} saniye/adet")
    st.write(f"**Günlük Maksimum Ürün (450 dk):** {max_daily_production} adet")

    # ======== VERİMLİLİK ÖNERİSİ ========
    optimal_fabric  = 7
    optimal_workers = 1
    optimal_speed   = 3

    X_optimal = build_feature_row(optimal_fabric, optimal_speed, optimal_workers, embroidery_type, expected_cols)
    try:
        predicted_optimal_time = float(prod_model.predict(X_optimal)[0])
    except Exception as e:
        st.warning(f"Optimum senaryo tahmininde hata (öneri gösterilmeyecek): {e}")
        predicted_optimal_time = predicted_total_time

    optimal_unit_time = predicted_optimal_time / optimal_fabric
    verim_artisi = ((unit_time - optimal_unit_time) / unit_time) * 100 if unit_time > 0 else 0.0

    st.subheader("🚦 Verimlilik Analizi")
    if fabric_count == optimal_fabric and num_workers == optimal_workers and machine_speed == optimal_speed:
        st.success("🎉 Bravo! Zaten en verimli kombinasyondasın.")
    elif verim_artisi > 0:
        st.warning(f"🚨 Potansiyelin altında çalışıyorsun! ~%{round(verim_artisi, 2)} artış fırsatı var.")
        st.markdown(
            f"Kumaş **{fabric_count} → {optimal_fabric}**, "
            f"İşçi **{num_workers} → {optimal_workers}**, "
            f"Makine hızı **{machine_speed} → {optimal_speed} dk** → **yaparsan daha verimli üretim elde edersin!**"
        )
    else:
        st.info(f"🔧 ~%{abs(round(verim_artisi, 2))} iyileşme için küçük dokunuşlar yapabilirsin.")
        st.markdown(
            f"Kumaş **{fabric_count} → {optimal_fabric}**, "
            f"İşçi **{num_workers} → {optimal_workers}**, "
            f"Makine hızı **{machine_speed} → {optimal_speed} dk** öneriyoruz."
        )
