# 🧵 Kizilay Shelter Systems: AI-Powered Production Predictor & Optimizer

This application is the interactive front-end of the **Kizilay Embroidery Efficiency Project**. Built with **Streamlit**, it allows production managers to predict manufacturing durations and receive AI-driven optimization suggestions in real-time.



## 🌟 Key Features

- **Real-Time Prediction:** Estimates total production time and unit time based on current machine settings.
- **Dynamic Optimization Engine:** Compares the user's input with the "Mathematically Optimal" scenario (derived from previous R-based analyses) and calculates the potential efficiency gain percentage.
- **Daily Capacity Forecasting:** Automatically calculates the maximum daily production output for a standard 450-minute shift.
- **Intelligent Feature Mapping:** Includes a robust alias-mapping system to handle various naming conventions from the underlying ML model (e.g., handling Turkish characters and different column names).

## 🛠️ Tech Stack

- **Framework:** [Streamlit](https://streamlit.io/) (Web UI)
- **Machine Learning:** `Scikit-learn` (via `joblib`)
- **Data Handling:** `Pandas`, `NumPy`
- **Styling:** Custom CSS/HTML injection for Kızılay branding.

## 📊 How the Logic Works

1. **Input:** The user enters the fabric count, worker count, machine speed, and embroidery type.
2. **ML Inference:** The app loads `model_ascii.pkl` and constructs a feature vector using the `build_feature_row` function.
3. **Benchmarking:** The system runs a second hidden inference using **Optimal Parameters** ($Fabric=7, Worker=1, Speed=3$).
4. **Actionable Insights:** It provides a visual alert (Success/Warning/Info) showing how much efficiency ($~X\%$) is being lost and what specific changes are needed to reclaim it.

## 🚀 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/kizilay-ai-predictor.git](https://github.com/yourusername/kizilay-ai-predictor.git)
Install dependencies:Bashpip install streamlit pandas numpy joblib
Place the Model: Ensure model_ascii.pkl and kizilay_logo.png are in the root directory.Launch the App:Bashstreamlit run app.py
Parameter,Range/Options,Description
Fabric Count,1+,Number of units in the current batch.
Machine Speed,1 - 6 min,Processing time setting on the machine.
Worker Count,1+,Number of operators assigned to the line.
Embroidery Type,Gri / Kırmızı / SAR,The specific design variant being produced.
