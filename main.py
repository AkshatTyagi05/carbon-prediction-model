import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from feature_engine.outliers import Winsorizer
from statsmodels.tsa.stattools import adfuller
import datetime

# --- CONFIG ---
st.set_page_config(page_title="Carbon AI v7.0", layout="wide")

# --- 1. ROBUST DATA ENGINE ---
@st.cache_data
# --- 1. ROBUST DATA ENGINE (VERIFIED ORDER) ---
@st.cache_data
# --- 1. DATA ENGINE (NUCLEAR OPTION FOR NaNs) ---
@st.cache_data
def load_and_clean():
    df = pd.read_csv('city_day.csv')
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    # 1. Select only the columns we need to clean
    cols = ['AQI', 'PM2.5', 'PM10', 'CO']
    
    # 2. Sort to ensure time-continuity
    df = df.sort_values('Datetime')

    # 3. Identify sensor errors and set to NaN
    for col in cols:
        if col in df.columns:
            df.loc[df[col] < 15, col] = np.nan
            
    # 4. TRIPLE-LAYER FILL (Essential to prevent the ValueError)
    # Layer 1: Linear interpolation for gaps
    df[cols] = df[cols].interpolate(method='linear', limit_direction='both')
    # Layer 2: Forward fill for gaps at the end
    df[cols] = df[cols].ffill()
    # Layer 3: Backward fill for gaps at the start
    df[cols] = df[cols].bfill()
    # Layer 4: Final safety check (Nuclear option)
    df[cols] = df[cols].fillna(0)

    # 5. NOW apply Winsorization (Safe because NaNs are 100% gone)
    try:
        wz = Winsorizer(capping_method="iqr", fold=1.5, tail="both")
        df[cols] = wz.fit_transform(df[cols])
    except Exception as e:
        st.warning(f"Note: Winsorization adjusted due to data distribution: {e}")

    # 6. Smooth the data for the 'Ups and Downs'
    for col in cols:
        df[col] = df[col].rolling(window=3, min_periods=1).mean()
        
    df['Year'] = df['Datetime'].dt.year
    return df

df = load_and_clean()

# --- 2. HEADER & UI FIX (Define variables early) ---
st.title("🌱 Carbon Emission Analysis and Forecasting")

all_cities = df['City'].unique()
# Ensure Delhi is default if it exists, otherwise take the first city
default_idx = list(all_cities).index('Delhi') if 'Delhi' in all_cities else 0

# --- 3. NATIONAL LEADERBOARD ---
top_10 = df.groupby('City')['AQI'].mean().sort_values(ascending=False).head(10).reset_index()
fig_leader = px.bar(top_10, x='AQI', y='City', orientation='h', title="National Pollution Leaderboard",
                   color='AQI', color_continuous_scale='Reds', template="plotly_dark")
st.plotly_chart(fig_leader, use_container_width=True)

st.divider()

# --- 4. STATE ANALYSIS ---
c1, c2 = st.columns([1, 3])
with c1:
    sel_state = st.selectbox("Select State/City", all_cities, index=default_idx)
    sel_metric = st.selectbox("Select Metric", ["AQI", "PM2.5", "PM10"])
    
    st.write("### 🩺 Data Health")
    state_vals = df[df['City'] == sel_state][sel_metric].values
    adf_p = adfuller(state_vals)[1]
    if adf_p < 0.05:
        st.success(f"Stationary (p={adf_p:.4f})")
    else:
        st.warning(f"Trend Detected (p={adf_p:.4f})")

with c2:
    state_df = df[df['City'] == sel_state].sort_values('Datetime')
    fig_hist = px.line(state_df.tail(400), x='Datetime', y=sel_metric, 
                      title=f"Historical {sel_metric} - {sel_state}",
                      template="plotly_dark", color_discrete_sequence=['#00ffcc'])
    st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# --- 5. PREDICTIVE ENGINE (FIXED FOR CURVES & YEARLY) ---
st.subheader("🔮 Predictive Forecasting Engine")
p1, p2 = st.columns([1, 2])

with p1:
    # --- THIS RESTORES THE MISSING DROPDOWN ---
    f_state = st.selectbox("Forecast Target", all_cities, key="f_target_dropdown", 
                           index=list(all_cities).index(sel_state))
    
    f_freq = st.radio("Interval", ["Daily", "Weekly", "Monthly", "Yearly"])
    f_steps = st.number_input("Forecast Steps", 1, 30, 3 if f_freq=="Yearly" else 7)
    run_btn = st.button("Generate AI Forecast", type="primary")

if run_btn:
    with st.spinner("🧠 AI Learning Temporal Waves..."):
        freq_map = {"Daily": 'D', "Weekly": 'W', "Monthly": 'M', "Yearly": 'Y'}
        p_data = df[df['City'] == sel_state].set_index('Datetime')['AQI'].resample(freq_map[f_freq]).mean().fillna(method='ffill')
        
        # 1. Feature Engineering (Circular encoding for Month AND Day)
        train_df = pd.DataFrame({'AQI': p_data.values})
        train_df['M_Sin'] = np.sin(2 * np.pi * p_data.index.month / 12)
        train_df['M_Cos'] = np.cos(2 * np.pi * p_data.index.month / 12)
        
        # Use Yeo-Johnson transform to make data more "Normal" (Fixes flat lines)
        pt = PowerTransformer(method='yeo-johnson')
        scaled = pt.fit_transform(train_df)
        
        window = 3 if f_freq == "Yearly" else 14
        X, y = [], []
        for i in range(len(scaled) - window):
            X.append(scaled[i:i+window])
            y.append(scaled[i+window, 0])
        
        # 2. 32-Unit Model (Slightly more complex to follow curves)
        model = Sequential([
            LSTM(32, return_sequences=True, input_shape=(window, 3)),
            Dropout(0.1),
            LSTM(16),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        # Increased epochs to 50 to prevent flat lines
        model.fit(np.array(X), np.array(y), epochs=50, verbose=0, batch_size=16)
        
        # 3. Prediction Loop
        last_w = scaled[-window:].reshape(1, window, 3)
        future_preds = []
        curr_date = p_data.index[-1]

        for i in range(f_steps):
            p = model.predict(last_w, verbose=0)[0,0]
            future_preds.append(p)
            
            if f_freq == "Daily": curr_date += pd.Timedelta(days=1)
            elif f_freq == "Weekly": curr_date += pd.Timedelta(weeks=1)
            elif f_freq == "Monthly": curr_date += pd.DateOffset(months=1)
            else: curr_date += pd.DateOffset(years=1)
            
            new_row = np.array([p, np.sin(2*np.pi*curr_date.month/12), np.cos(2*np.pi*curr_date.month/12)]).reshape(1,1,3)
            last_w = np.append(last_w[:, 1:, :], new_row, axis=1)
            
        # 4. Inverse Transform
        res_dummy = np.zeros((len(future_preds), 3))
        res_dummy[:, 0] = future_preds
        res = pt.inverse_transform(res_dummy)[:, 0]

        with p2:
            f_dates = pd.date_range(p_data.index[-1], periods=f_steps+1, freq=freq_map[f_freq])[1:]
            conn_dates = np.insert(f_dates, 0, p_data.index[-1])
            conn_preds = np.insert(res, 0, p_data.values[-1])

            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=p_data.index[-60:], y=p_data.values[-60:], name="Past"))
            fig_f.add_trace(go.Scatter(x=conn_dates, y=conn_preds, name="AI Forecast", 
                                     line=dict(dash='dash', color='#ef4444', width=3), mode='lines+markers'))
            fig_f.update_layout(template="plotly_dark", title=f"Natural Curve LSTM Forecast")
            st.plotly_chart(fig_f, use_container_width=True)
            st.table(pd.DataFrame({"Date": f_dates.strftime('%Y-%m-%d'), "AQI": res.astype(int)}))

st.markdown("---")
st.caption("v7.0 | Advanced Power Transformation & Multi-Layer LSTM")