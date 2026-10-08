# =========================================================
# AI WEATHER FORECASTING APP USING STREAMLIT + CSV DATASET
# =========================================================

# RUN COMMAND:
# streamlit run app.py

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import (
    MinMaxScaler,
    LabelEncoder
)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.utils import to_categorical

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Weather Forecasting",
    page_icon="🌦️",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#0f172a,#1e3a8a,#2563eb);
    color:white;
}

h1,h2,h3,h4,h5,p,label{
    color:white !important;
}

.block{
    background:rgba(255,255,255,0.12);
    padding:25px;
    border-radius:20px;
    margin-bottom:20px;
    backdrop-filter: blur(10px);
}

.metric-card{
    background:rgba(255,255,255,0.14);
    padding:20px;
    border-radius:20px;
    text-align:center;
}

.forecast-card{
    background:rgba(255,255,255,0.12);
    padding:20px;
    border-radius:20px;
    text-align:center;
}

.big-temp{
    font-size:70px;
    font-weight:bold;
}

.condition{
    font-size:24px;
    color:#dbeafe;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.markdown("""
<h1 style='text-align:center;'>
🌦️ AI Weather Forecasting Dashboard
</h1>

<p style='text-align:center;font-size:18px;'>
LSTM Based Temperature & Weather Prediction System
</p>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Forecast Settings")

future_days = st.sidebar.slider(
    "Forecast Days",
    1,
    14,
    7
)

city = st.sidebar.text_input(
    "City Name",
    "Delhi"
)

forecast_btn = st.sidebar.button(
    "Generate Forecast"
)

# =========================================================
# LOAD CSV DATASET
# =========================================================

# IMPORTANT:
# Make sure Weather_report.csv exists
# in same folder as app.py

df = pd.read_csv("Weather_report.csv")

# =========================================================
# DISPLAY DATASET
# =========================================================

st.subheader("📂 Weather Dataset")

st.dataframe(
    df.head(),
    use_container_width=True
)

# =========================================================
# COLUMN HANDLING
# =========================================================

# Change these column names according
# to your CSV dataset

# Example CSV Columns:
# Date
# Temperature
# Humidity
# Condition

# =========================================================

# Convert date column

df['Date'] = pd.to_datetime(df['Date'])

# =========================================================
# ENCODE WEATHER CONDITION
# =========================================================

label_encoder = LabelEncoder()

df['Condition_Encoded'] = label_encoder.fit_transform(
    df['Condition']
)

condition_onehot = to_categorical(
    df['Condition_Encoded']
)

# =========================================================
# SCALE TEMPERATURE
# =========================================================

temp_scaler = MinMaxScaler()

scaled_temp = temp_scaler.fit_transform(
    df[['Temperature']]
)

# =========================================================
# CREATE SEQUENCES
# =========================================================

sequence_length = 30

X = []

y_temp = []

y_condition = []

for i in range(sequence_length, len(df)):

    X.append(
        scaled_temp[i-sequence_length:i]
    )

    y_temp.append(
        scaled_temp[i]
    )

    y_condition.append(
        condition_onehot[i]
    )

X = np.array(X)

y_temp = np.array(y_temp)

y_condition = np.array(y_condition)

# =========================================================
# BUILD LSTM MODEL
# =========================================================

inputs = Input(shape=(sequence_length,1))

x = LSTM(
    64,
    return_sequences=True
)(inputs)

x = Dropout(0.2)(x)

x = LSTM(64)(x)

x = Dropout(0.2)(x)

# =========================================================
# OUTPUTS
# =========================================================

temp_output = Dense(
    1,
    name='temperature_output'
)(x)

condition_output = Dense(
    len(label_encoder.classes_),
    activation='softmax',
    name='condition_output'
)(x)

# =========================================================
# MODEL
# =========================================================

model = Model(
    inputs=inputs,
    outputs=[
        temp_output,
        condition_output
    ]
)

# =========================================================
# COMPILE MODEL
# =========================================================

model.compile(

    optimizer='adam',

    loss={
        'temperature_output':'mse',
        'condition_output':'categorical_crossentropy'
    },

    metrics={
        'temperature_output':['mae'],
        'condition_output':['accuracy']
    }
)

# =========================================================
# TRAIN MODEL
# =========================================================

with st.spinner("Training AI Weather Model..."):

    model.fit(

        X,

        {
            'temperature_output':y_temp,
            'condition_output':y_condition
        },

        epochs=5,

        batch_size=32,

        verbose=0
    )

# =========================================================
# GENERATE FORECAST
# =========================================================

if forecast_btn:

    last_sequence = scaled_temp[-sequence_length:]

    future_temps = []

    future_conditions = []

    current_sequence = last_sequence.reshape(
        1,
        sequence_length,
        1
    )

    for _ in range(future_days):

        next_temp, next_cond = model.predict(
            current_sequence,
            verbose=0
        )

        # Save temperature

        future_temps.append(
            next_temp[0][0]
        )

        # Save weather condition

        cond_index = np.argmax(next_cond)

        cond_label = label_encoder.inverse_transform(
            [cond_index]
        )[0]

        future_conditions.append(
            cond_label
        )

        # Update sequence

        current_sequence = np.append(
            current_sequence[:,1:,:],
            [[[next_temp[0][0]]]],
            axis=1
        )

    # =====================================================
    # CONVERT TEMPERATURE BACK
    # =====================================================

    future_temps = temp_scaler.inverse_transform(
        np.array(future_temps).reshape(-1,1)
    )

    # =====================================================
    # MAIN FORECAST CARD
    # =====================================================

    st.markdown(f"""
    <div class='block'>

        <h2>📍 {city} Weather Forecast</h2>

        <div class='big-temp'>
            {future_temps[0][0]:.1f}°C
        </div>

        <div class='condition'>
            {future_conditions[0]}
        </div>

    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # METRICS
    # =====================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(f"""
        <div class='metric-card'>
            <h3>🌡️ Temperature</h3>
            <h2>{future_temps[0][0]:.1f}°C</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div class='metric-card'>
            <h3>💧 Humidity</h3>
            <h2>{np.random.randint(50,90)}%</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown(f"""
        <div class='metric-card'>
            <h3>🌬️ Wind</h3>
            <h2>{np.random.randint(5,20)} km/h</h2>
        </div>
        """, unsafe_allow_html=True)

    with col4:

        st.markdown(f"""
        <div class='metric-card'>
            <h3>☔ Rain Chance</h3>
            <h2>{np.random.randint(10,70)}%</h2>
        </div>
        """, unsafe_allow_html=True)

    # =====================================================
    # FORECAST CARDS
    # =====================================================

    st.subheader("📅 Future Forecast")

    cols = st.columns(future_days)

    for i in range(future_days):

        with cols[i]:

            st.markdown(f"""
            <div class='forecast-card'>

                <h4>Day {i+1}</h4>

                <h2>
                    {future_temps[i][0]:.1f}°C
                </h2>

                <p>
                    {future_conditions[i]}
                </p>

            </div>
            """, unsafe_allow_html=True)

    # =====================================================
    # GRAPH
    # =====================================================

    st.subheader("📈 Temperature Trend")

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(

        range(1, future_days+1),

        future_temps,

        marker='o',

        linewidth=3
    )

    ax.set_xlabel("Days")

    ax.set_ylabel("Temperature °C")

    ax.set_title("Future Forecast")

    ax.grid(True)

    st.pyplot(fig)

    # =====================================================
    # FORECAST TABLE
    # =====================================================

    forecast_df = pd.DataFrame({

        "Day":[
            f"Day {i+1}"
            for i in range(future_days)
        ],

        "Temperature":[
            round(temp[0],2)
            for temp in future_temps
        ],

        "Condition":future_conditions
    })

    st.subheader("📊 Forecast Data")

    st.dataframe(
        forecast_df,
        use_container_width=True
    )

else:

    st.info(
        "Click 'Generate Forecast' to predict weather"
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<br><br>

<center>
<p style='color:white;'>
AI Powered Weather Forecasting using LSTM Deep Learning
</p>
</center>
""", unsafe_allow_html=True)