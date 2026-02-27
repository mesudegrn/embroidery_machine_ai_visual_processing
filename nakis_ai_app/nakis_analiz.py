# VSCode'da çalıştırılabilir hale getirmek için streamlit yerine tkinter veya OpenCV gibi arayüzler tercih edilebilir.
# Ancak, bu örnekte sadece kodun Streamlit bağımlılıklarından kurtarılmış halini oluşturacağım.
# Kullanıcı arayüzü yerine terminal üzerinden giriş alıp tahmin sonuçlarını gösterecek şekilde düzenlenecek.

import pandas as pd
import joblib
import math
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# MODELLERİ YÜKLE
img_model = load_model("keras_model.h5")
prod_model = joblib.load("model_ascii.pkl")

class_names = ["gri_arma", "kirmizi_arma", "SAR_arma"]

# Görüntü dosyasını al
def predict_image_class(image_path):
    image = Image.open(image_path).convert("RGB")
    img_resized = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
    prediction = img_model.predict(img_array)
    predicted_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_index]
    predicted_percent = prediction[0][predicted_index] * 100
    return predicted_class, predicted_percent

# Kolon temizleyici
def clean_columns(df):
    df.columns = df.columns.str.normalize('NFKD')\
                           .str.encode('ascii', errors='ignore')\
                           .str.decode('utf-8')\
                           .str.replace('[^A-Za-z0-9_]+', '_', regex=True)\
                           .str.lower()
    return df

# Tahmin fonksiyonu
def run_prediction(image_path, fabric_count, machine_speed, num_workers):
    predicted_class, percent = predict_image_class(image_path)
    print(f"\n🧠 Tespit Edilen Nakış Türü: {predicted_class.upper()} ({percent:.2f}%)\n")

    # One-hot encoding
    embroidery_type_gri_arma = 1 if predicted_class == "gri_arma" else 0
    embroidery_type_kirmizi_arma = 1 if predicted_class == "kirmizi_arma" else 0
    embroidery_type_sar_arma = 1 if predicted_class == "SAR_arma" else 0

    X_input = pd.DataFrame([{
        "fabric_count": fabric_count,
        "machine_time_min": machine_speed,
        "num_of_workers": num_workers,
        "embroidery_type_gri_arma": embroidery_type_gri_arma,
        "embroidery_type_kirmizi_arma": embroidery_type_kirmizi_arma,
        "embroidery_type_sar_arma": embroidery_type_sar_arma
    }])

    X_input = clean_columns(X_input)
    predicted_total_time = prod_model.predict(X_input)[0]
    unit_time = predicted_total_time / fabric_count
    max_daily_production = math.floor((450 * 60) / unit_time)

    print(f"⏱️ Tahmini Toplam Süre: {round(predicted_total_time, 2)} saniye")
    print(f"⏱️ Birim Kumaş Süresi: {round(unit_time, 2)} saniye")
    print(f"🏭 Günlük Maksimum Üretim: {max_daily_production} adet\n")

    # Verimlilik önerisi
    improved_fabric = 8
    improved_workers = 1
    improved_speed = 3

    X_scenario = pd.DataFrame([{
        "fabric_count": improved_fabric,
        "machine_time_min": improved_speed,
        "num_of_workers": improved_workers,
        "embroidery_type_gri_arma": embroidery_type_gri_arma,
        "embroidery_type_kirmizi_arma": embroidery_type_kirmizi_arma,
        "embroidery_type_sar_arma": embroidery_type_sar_arma
    }])

    X_scenario = clean_columns(X_scenario)
    predicted_scenario_time = prod_model.predict(X_scenario)[0]
    new_unit_time = predicted_scenario_time / improved_fabric
    verim_artisi = ((unit_time - new_unit_time) / unit_time) * 100

    if verim_artisi < 5:
        print("🎉 Tebrikler! Şu an zaten en verimli moddasınız.")
    else:
        print("🚀 Verimlilik Önerisi:")
        print(f"Kumaş sayısını {fabric_count} → {improved_fabric},")
        print(f"İşçi sayısını {num_workers} → {improved_workers},")
        print(f"Makine hızını {machine_speed} dk → {improved_speed} dk yaparsanız...")
        print(f"💥 %{round(verim_artisi, 2)} daha fazla verim elde edersiniz!\n")
