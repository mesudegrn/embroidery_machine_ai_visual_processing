import streamlit as st
import pandas as pd
import joblib
import math
import base64
import numpy as np
from PIL import Image


import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import load_model

# Logo için CSS ve HTML
st.markdown(
    """
    <style>
    .kizilay-logo-container {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: -1;
        opacity: 0.15;
    }

    .kizilay-logo-container img {
        width: 220px;
    }

    .kizilay-baslik {
        font-size: 50px;
        color: red;
        font-weight: bold;
        text-align: center;
        margin-top: 10px;
    }

    .kizilay-subtitle {
        font-size: 26px;
        color: #555;
        text-align: center;
        margin-bottom: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Logo görüntüsünü base64 ile gömme
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

kizilay_logo_base64 = get_base64_of_bin_file("kizilay_logo.png")

# HTML olarak gömme
st.markdown(
    f"""
    <div class="kizilay-logo-container">
        <img src="data:image/png;base64,{kizilay_logo_base64}">
    </div>
    <div class="kizilay-baslik">KIZILAY BARINMA SİSTEMLERİ</div>
    <div class="kizilay-subtitle">🧵 Nakış Üretim Süresi ve Verimlilik Tahmini</div>
    """,
    unsafe_allow_html=True
)

# === MODELLERİ YÜKLE ===
img_model = load_model("keras_model.h5")  # Teachable Machine modeli
prod_model = joblib.load("model_ascii.pkl")        # Üretim süresi tahmin modeli

class_names = ["gri_arma", "kirmizi_arma", "SAR_arma"]  # Teachable'daki sıralamaya göre

st.title("Kameradan Nakış Türü ve Üretim Süresi Tahmini")



# st.markdown(
#     """
#     <style>
#     /* Logo sadece arka planda ve yarı opak */
#     .logo-background {
#         position: fixed;
#         top: 20px;
#         right: 20px;
#         width: 300px;
#         opacity: 0.15;
#         z-index: -1;
#     }

#     /* Kırmızı büyük başlık */
#     .kizilay-baslik {
#         font-size: 50px;
#         color: red;
#         font-weight: bold;
#         text-align: center;
#         margin-top: 10px;
#     }

#     /* Gri alt başlık */
#     .kizilay-subtitle {
#         font-size: 26px;
#         color: #555;
#         text-align: center;
#         margin-bottom: 40px;
#     }
#     </style>

#     <!-- Arka plandaki logo -->
#     <img class="logo-background" src="logo-turk-kizilay.png">

#     <!-- Başlıklar -->
#     <div class="kizilay-baslik">KIZILAY BARINMA SİSTEMLERİ</div>
#     <div class="kizilay-subtitle"> Nakış Üretim Süresi ve Verimlilik Tahmini</div>
#     """,
#     unsafe_allow_html=True
# )





# Temizlik fonksiyonu
def clean_columns(df):
    df.columns = df.columns.str.normalize('NFKD')\
                           .str.encode('ascii', errors='ignore')\
                           .str.decode('utf-8')\
                           .str.replace('[^A-Za-z0-9_]+', '_', regex=True)\
                           .str.lower()
    return df

st.title(" Nakış Üretim Süresi ve Verimlilik Tahmini")



img_file = st.camera_input("📷 Nakış Görselini Yükle")

if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    img_resized = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    # === GÖRÜNTÜ MODELİYLE SINIFLANDIR ===
    prediction = img_model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction[0])]
    st.success(f"🧠 Tespit Edilen Nakış Türü: **{predicted_class}**")

    # === ONE-HOT ENCODING ===
    embroidery_type_gri_arma = 1 if predicted_class == "gri_arma" else 0
    embroidery_type_kirmizi_arma = 1 if predicted_class == "kirmizi_arma" else 0
    embroidery_type_sar_arma = 1 if predicted_class == "SAR_arma" else 0



# 🔧 GİRİŞLER
fabric_count = st.number_input("Kumaş Sayısı", min_value=1)
machine_speed = st.number_input("Makine Hızı (dk)", min_value=1)
num_workers = st.number_input("İşçi Sayısı", min_value=1)
#embroidery_type = st.selectbox("Nakış Türü", ["gri_arma", "kirmizi_arma"])

# Nakış türü kodlama
#embroidery_type_gri_arma = 1 if embroidery_type == "gri_arma" else 0



# Model yükle
model = joblib.load("model_ascii.pkl")



X_input = pd.DataFrame([{
    "fabric_count": fabric_count,
    "machine_time_min": machine_speed,
    "num_of_workers": num_workers,
    "embroidery_type_gri_arma": embroidery_type_gri_arma,
    "embroidery_type_kirmizi_arma": embroidery_type_kirmizi_arma,
    "embroidery_type_sar_arma": embroidery_type_sar_arma
}])






# BUTON EKLE
if st.button("🔍 Tahmin Et"):

    # ✨ Tahmin hesaplamaları
    X_input = pd.DataFrame([{
        "fabric_count": fabric_count,
        "machine_time_min": machine_speed,
        "num_of_workers": num_workers,
        "embroidery_type_gri_arma": embroidery_type_gri_arma
    }])
    X_input = clean_columns(X_input)

    # Tahmini toplam süre
    predicted_total_time = model.predict(X_input)[0]
    unit_time = predicted_total_time / fabric_count
    max_daily_production = math.floor((450 * 60) / unit_time)

    # Tahmin sonuçları
    st.subheader("⏱️ Tahmin Sonuçları")
    st.write(f"**Tahmini Toplam Süre:** {round(predicted_total_time, 2)} saniye")
    st.write(f"**Birim Kumaş Süresi:** {round(unit_time, 2)} saniye")
    st.write(f"**Günlük Maksimum Ürün:** {max_daily_production} adet")

# # 🧠 VERİMLİLİK ÖNERİSİ
improved_fabric = 8
improved_workers = 1
improved_speed = 3


X_scenario = pd.DataFrame([{
        "fabric_count": improved_fabric,
        "machine_time_min": improved_speed,
        "num_of_workers": improved_workers,
        "embroidery_type_gri_arma": embroidery_type_gri_arma
    }])


X_scenario = clean_columns(X_scenario)
predicted_scenario_time = model.predict(X_scenario)[0]
new_unit_time = predicted_scenario_time / improved_fabric
verim_artisi = ((unit_time - new_unit_time) / unit_time) * 100

if verim_artisi < 5:
        st.success("🎉 Tebrikler! Şu an zaten en verimli moddasınız.")
else:
        st.subheader("🚀 Verimlilik Önerisi")
        st.markdown(f"""
                        Kumaş sayısını **{fabric_count} → {improved_fabric}**,  
                        işçi sayısını **{num_workers} → {improved_workers}**,  
                        makine hızını **{machine_speed} → {improved_speed} dk** yaparsanız...  
                        **💥 %{round(verim_artisi, 2)} daha fazla verim elde edersiniz!**
                        """)









# # Tahmin girişi
# X_input = pd.DataFrame([{
#     "fabric_count": fabric_count,
#     "machine_time_min": machine_speed,
#     "num_of_workers": num_workers,
#     "embroidery_type_gri_arma": embroidery_type_gri_arma
# }])
# X_input = clean_columns(X_input)

# # Tahmini toplam süre
# predicted_total_time = model.predict(X_input)[0]
# unit_time = predicted_total_time / fabric_count
# max_daily_production = math.floor((450 * 60) / unit_time)  # 450 dk = 27000 saniye

# # ✨ ÇIKTILAR
# st.subheader("⏱️ Tahmin Sonuçları")
# st.write(f"**Tahmini Toplam Süre:** {round(predicted_total_time, 2)} saniye")
# st.write(f"**Birim Kumaş Süresi:** {round(unit_time, 2)} saniye")
# st.write(f"**Günlük Maksimum Ürün:** {max_daily_production} adet")

# # 🧠 VERİMLİLİK ÖNERİSİ
# improved_fabric = 8
# improved_workers = 1
# improved_speed = 3

# X_scenario = pd.DataFrame([{
#     "fabric_count": improved_fabric,
#     "machine_time_min": improved_speed,
#     "num_of_workers": improved_workers,
#     "embroidery_type_gri_arma": embroidery_type_gri_arma
# }])
# X_scenario = clean_columns(X_scenario)
# predicted_scenario_time = model.predict(X_scenario)[0]
# new_unit_time = predicted_scenario_time / improved_fabric

# # Verim hesapla
# verim_artisi = ((unit_time - new_unit_time) / unit_time) * 100

# # ✨ VERİMLİLİK ÖNERİSİ GÖSTER
# st.subheader("🚀 Verimlilik Önerisi")
# st.markdown(f"""
# Kumaş sayısını **{fabric_count} → {improved_fabric}**,  
# işçi sayısını **{num_workers} → {improved_workers}**,  
# makine hızını **{machine_speed} → {improved_speed} dk** yaparsanız...  
# **💥 %{round(verim_artisi, 2)} daha fazla verim elde edersiniz!**
# """)
