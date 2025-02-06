import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# 📌 Set page config
st.set_page_config(
    page_title="📱 Mobile Price Classification",
    page_icon="📱",
    layout="wide"
)

# 📌 Function to Load Model & Scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        with open('LogisticRegression.pkl', 'rb') as file:
            model = pickle.load(file)
        if not hasattr(model, 'predict'):
            st.error('⚠️ The loaded model is not valid. Ensure the correct model is being loaded.')
            st.stop()
        with open('StandardScaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"⚠️ Error: {e}. Ensure model and scaler exist in the correct directory.")
        st.stop()

# Load the model and scaler
model, scaler = load_model_and_scaler()

# 📌 Title and description
st.title("📱 Mobile Price Classification (Updated for 2024)")
st.markdown("""
This app predicts the price range of a mobile phone based on its specifications.
* **Price Ranges:** 0 (Low Cost), 1 (Medium Cost), 2 (High Cost), 3 (Very High Cost)
""")

# 📌 Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📲 Device Specifications (Modernized)")

    # 📌 Updated 2024 Mobile Specifications
    input_col1, input_col2, input_col3 = st.columns(3)
    
    with input_col1:
        battery = st.slider("🔋 Battery Power (mAh)", 1500, 6000, 4000)  # Modern battery range
        clock_speed = st.slider("⚡ Clock Speed (GHz)", 1.0, 3.5, 2.0)
        dual_sim = st.selectbox("📶 Dual SIM", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        front_camera = st.slider("🤳 Front Camera (MP)", 5, 64, 16)
        int_memory = st.slider("💾 Internal Storage (GB)", 16, 512, 128)  # Modern storage options

    with input_col2:
        primary_camera = st.slider("📷 Rear Camera (MP)", 8, 200, 64)  # Modernized range
        ram = st.slider("💨 RAM (MB)", 1024, 16000, 8000)  # Up to 16GB
        touch_screen = st.selectbox("🖥️ Touch Screen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        wifi = st.selectbox("📡 WiFi", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        five_g = st.selectbox("🚀 5G Enabled", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    with input_col3:
        cores = st.slider("🛠️ CPU Cores", 2, 12, 6)  # More powerful CPUs
        refresh_rate = st.slider("🔄 Refresh Rate (Hz)", 60, 165, 120)  # New feature
        fast_charging = st.slider("⚡ Charging Speed (W)", 10, 120, 30)  # New feature
        ai_features = st.selectbox("🤖 AI Features", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    # 📌 Adjust Values to Match Old Dataset Scale
    scaled_ram = ram / 4  # Reducing modern RAM values to match old dataset
    scaled_int_memory = int_memory / 4  # Same for storage
    scaled_battery = battery / 2  # Scaling down to match dataset

    # 📌 Create input dataframe
    input_data = {
        'Battery Power': scaled_battery,
        'Clock speed': clock_speed,
        'Dual SIM': dual_sim,
        'Front Camera': front_camera,
        '5G': five_g,
        'Internal Memory': scaled_int_memory,
        'Primary Camera': primary_camera,
        'Ram': scaled_ram,
        'Touch Screen': touch_screen,
        'WiFi': wifi,
        'CPU Cores': cores,
        'Refresh Rate': refresh_rate / 10,  # Adjusting scale
        'Fast Charging': fast_charging / 10,  # Adjusting scale
        'AI Features': ai_features
    }

    input_df = pd.DataFrame([input_data])

    # 📌 Scale input data before prediction
    input_df_scaled = scaler.transform(input_df)

    # 📌 Add a predict button
    if st.button("🚀 Predict Price Range"):
        prediction = model.predict(np.array(input_df_scaled).reshape(1, -1))[0]
        price_ranges = {0: "Low Cost", 1: "Medium Cost", 2: "High Cost", 3: "Very High Cost"}
        st.success(f"🎯 Predicted Price Range: **{price_ranges[prediction]}**")

        # 📌 Display prediction probabilities
        probabilities = model.predict_proba(input_df_scaled)[0]
        prob_df = pd.DataFrame({
            'Price Range': list(price_ranges.values()),
            'Probability': probabilities * 100
        })

        # 📊 Use Plotly for Better Graphs
        fig = px.bar(prob_df, x="Price Range", y="Probability", title="Prediction Probabilities",
                     text=prob_df['Probability'].round(2),
                     color="Price Range", color_discrete_sequence=["blue", "green", "orange", "red"])
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig)

# 📌 Sidebar for Additional Info
st.sidebar.header("📌 About This App")
st.sidebar.info("""
📲 **Modernized Mobile Price Classification App**  
🔍 Uses **Logistic Regression** to classify mobile price range.  
📊 **Updated for 2024 specifications**  
⚡ **Now includes 5G, AI Features, Fast Charging, Refresh Rate**
""")

st.sidebar.header("📊 Model Performance")
st.sidebar.markdown("""
- **Accuracy:** 83.0%
- **Algorithm Used:** Logistic Regression
""")
