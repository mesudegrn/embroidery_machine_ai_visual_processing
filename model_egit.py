import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

def clean_columns(df):
    df.columns = df.columns.str.normalize('NFKD')\
                           .str.encode('ascii', errors='ignore')\
                           .str.decode('utf-8')\
                           .str.replace('[^A-Za-z0-9_]+', '_', regex=True)\
                           .str.lower()
    return df

# Excel oku
df = pd.read_excel("Book1.xlsx")
df = clean_columns(df)

# Gerekli sütunları al
df = df[["fabric_count", "machine_time_min", "num_of_workers", "embroidery_type", "total_time"]]

# One-hot encoding
df = pd.get_dummies(df, columns=["embroidery_type"], drop_first=False)

# X ve y
X = df[["fabric_count", "machine_time_min", "num_of_workers", "embroidery_type_gri_arma"]]
y = df["total_time"]

# Model eğit
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Kaydet
joblib.dump(model, "model_ascii.pkl")
print("MODEL YENİDEN EĞİTİLDİ.")
