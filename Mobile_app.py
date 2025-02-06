import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# 📌 Set page config
st.set_page_config(
    page_title="Mobile Price Classification",
    page_icon="📱",
    layout="wide"
)

# 📌 Function to Load Model & Scaler
@st.cache_data
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
st.title("📱 Mobile Price Classification App")
st.markdown("""
This app predicts the price range of a mobile phone based on its specifications.
* **Price Ranges:** 0 (Low Cost), 1 (Medium Cost), 2 (High Cost), 3 (Very High Cost)
""")

# 📌 Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Device Specifications")
    
    # Create three columns for inputs
    input_col1, input_col2, input_col3 = st.columns(3)
    
    with input_col1:
        battery = st.slider("Battery Power (mAh)", 500, 2000, 1000)
        clock_speed = st.slider("Clock Speed (GHz)", 0.5, 3.0, 1.5)
        dual_sim = st.selectbox("Dual SIM", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        front_camera = st.slider("Front Camera (MP)", 0, 20, 5)
        int_memory = st.slider("Internal Memory (GB)", 2, 64, 32)
        
    with input_col2:
        primary_camera = st.slider("Primary Camera (MP)", 0, 20, 13)
        ram = st.slider("RAM (MB)", 256, 3998, 2048)
        touch_screen = st.selectbox("Touch Screen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        wifi = st.selectbox("WiFi", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        four_g = st.selectbox("4G", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    with input_col3:
        cores = st.slider("Number of Cores", 1, 8, 4)
        pixel_density = (1000 * 1200) / (12 * 7)  # Assuming avg screen size
        weight_screen_ratio = 140 / (12 * 7)  # Assuming avg weight of 140g
        talk_battery_ratio = 10 / battery  # Assuming avg talk time of 10 hours
        compute_power = (clock_speed * 1.5) + (cores * 1.2)
    
    # 📌 Create input dataframe
    input_data = {
        'Battery Power': battery,
        'Clock speed': clock_speed,
        'Dual SIM': dual_sim,
        'Front Camera': front_camera,
        '4G': four_g,
        'Internal Memory': int_memory,
        'Primary Camera': primary_camera,
        'Ram': ram,
        'Touch Screen': touch_screen,
        'wifi': wifi,
        'Pixel Density': pixel_density,
        'Weight Screen Ratio': weight_screen_ratio,
        'Talk Time Battery Ratio': talk_battery_ratio,
        'Compute Power': compute_power
    }

    input_df = pd.DataFrame([input_data])
    
    # 📌 Scale input data before prediction
    input_df_scaled = scaler.transform(input_df)
    
    # 📌 Add a predict button
    if st.button("Predict Price Range"):
        prediction = model.predict(np.array(input_df_scaled).reshape(1, -1))[0]
        price_ranges = {0: "Low Cost", 1: "Medium Cost", 2: "High Cost", 3: "Very High Cost"}
        st.success(f"Predicted Price Range: **{price_ranges[prediction]}**")

        # 📌 Display prediction probabilities
        probabilities = model.predict_proba(input_df_scaled)[0]
        prob_df = pd.DataFrame({
            'Price Range': list(price_ranges.values()),
            'Probability': probabilities * 100
        })
        
        # 📌 Plot with Matplotlib instead of Plotly
        fig, ax = plt.subplots()
        ax.bar(prob_df['Price Range'], prob_df['Probability'], color='blue')
        ax.set_xlabel("Price Range")
        ax.set_ylabel("Probability (%)")
        ax.set_title("Prediction Probabilities")
        st.pyplot(fig)

# 📌 Add "About" Section in Sidebar
st.sidebar.header("📌 About")
st.sidebar.info("""
This application predicts the price range of a mobile phone based on its specifications using **Logistic Regression**.

### **Model Information**
- **Algorithm:** Logistic Regression
- **Accuracy:** 83.0%
- **Trained on:** Scaled Mobile Phone Data

### **Features Considered**
- **Hardware Specs:** RAM, battery, processor
- **Camera:** Primary & Front camera
- **Connectivity:** WiFi, Bluetooth, 4G, Dual SIM
- **Physical Attributes:** Screen size, weight, mobile depth
""")

# 📌 Add model performance metrics
st.sidebar.header("📊 Model Performance")
st.sidebar.markdown("""
- **Accuracy:** 83.0%
- **Algorithm Used:** Logistic Regression
""")
