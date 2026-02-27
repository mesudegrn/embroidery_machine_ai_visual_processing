from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

model = joblib.load("model_ascii.pkl")

def clean_columns(df):
    df.columns = df.columns.str.normalize('NFKD')\
        .str.encode('ascii', errors='ignore')\
        .str.decode('utf-8')\
        .str.replace('[^A-Za-z0-9_]+', '_', regex=True)\
        .str.lower()
    return df

@app.route('/tahmin', methods=['POST'])
def tahmin():
    data = request.json
    df = pd.DataFrame([data])
    df.columns = (
        df.columns.str.normalize('NFKD')
                  .str.encode('ascii', errors='ignore')
                  .str.decode('utf-8')
                  .str.replace(r'[^A-Za-z0-9_]+', '_', regex=True)
                  .str.lower()
    )
    prediction = model.predict(df)
    total_time = prediction[0]
    unit_time = round(total_time / data["fabric_count"], 2)
    daily_max = round((60 * 8 * data["num_of_workers"]) / unit_time)

    return jsonify({
        "total_time": round(total_time, 2),
        "unit_time": unit_time,
        "daily_max": daily_max
    })

if __name__ == '__main__':
    app.run(debug=True)
