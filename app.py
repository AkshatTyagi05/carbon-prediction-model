import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from feature_engine.outliers import Winsorizer
from statsmodels.tsa.stattools import adfuller
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

st.set_page_config(page_title="Urban Air Intelligence System", layout="wide")

# =========================
# DATA LOADING
# =========================
@st.cache_data
def load_air_data():
    df = pd.read_csv('city_day.csv')
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df = df.dropna(subset=['Datetime']).sort_values('Datetime')

    cols = ['AQI', 'PM2.5', 'PM10']

    for col in cols:
        df.loc[df[col] < 0, col] = np.nan

    df[cols] = df[cols].interpolate().ffill().bfill().fillna(0)

    clean_cols = [col for col in cols if df[col].nunique() > 10]
    if len(clean_cols) > 0:
        try:
            wz = Winsorizer(capping_method="iqr", fold=1.5)
            df[clean_cols] = wz.fit_transform(df[clean_cols])
        except:
            wz = Winsorizer(capping_method="gaussian", fold=3)
            df[clean_cols] = wz.fit_transform(df[clean_cols])

    for col in cols:
        df[col] = df[col].rolling(3, min_periods=1).mean()

    df['Year'] = df['Datetime'].dt.year
    return df

@st.cache_data
def load_co2_data():
    return pd.read_csv('CO2 emissions India.csv')

df = load_air_data()
co2_df = load_co2_data()

# =========================
# HEADER
# =========================
st.title("🌍 Urban Air Intelligence System (UAIS)")
st.caption("UAIS v2.0 | AI-powered Environmental Intelligence & Decision Support System")

# =========================
# KPIs
# =========================
c1, c2, c3 = st.columns(3)
c1.metric("Avg AQI", int(df['AQI'].mean()))
c2.metric("Worst City", df.groupby('City')['AQI'].mean().idxmax())
c3.metric("Best City", df.groupby('City')['AQI'].mean().idxmin())

# =========================
# CITY SELECTION
# =========================
all_cities = sorted(df['City'].dropna().unique())
sel_city = st.selectbox("Select City", all_cities)

state_df = df[df['City'] == sel_city].sort_values('Datetime')

# =========================
# HEALTH + ALERT
# =========================
def get_health_advice(aqi):
    if aqi <= 50: return "🟢 Good"
    elif aqi <= 100: return "🟡 Moderate"
    elif aqi <= 200: return "🟠 Poor"
    elif aqi <= 300: return "🔴 Very Poor"
    else: return "⚫ Hazardous"

st.subheader("🌱 Health Advisory")
latest_aqi = state_df['AQI'].iloc[-1]
st.info(get_health_advice(latest_aqi))

st.subheader("🚨 Live Pollution Alert")
if latest_aqi > 300:
    st.error("🚨 Severe Pollution! Avoid outdoor exposure.")
elif latest_aqi > 200:
    st.warning("⚠️ Very Poor Air Quality")
elif latest_aqi > 100:
    st.info("🟠 Moderate Pollution")
else:
    st.success("✅ Air Quality Acceptable")

# =========================
# AQI GRAPH
# =========================
fig = px.line(state_df.tail(300), x='Datetime', y='AQI', title="AQI Trend")
st.plotly_chart(fig, use_container_width=True)

# =========================
# TREND INSIGHT
# =========================
st.subheader("📈 Trend Insights")
recent = state_df['AQI'].tail(30).mean()
past = state_df['AQI'].head(30).mean()

if recent > past:
    st.warning("📈 Pollution Increasing")
else:
    st.success("📉 Pollution Improving")

# =========================
# ANOMALY
# =========================
st.subheader("🚨 Pollution Spikes")
state_df['rolling'] = state_df['AQI'].rolling(7).mean()
state_df['anomaly'] = state_df['AQI'] > state_df['rolling'] * 1.5
st.dataframe(state_df[state_df['anomaly']].tail(10))

# =========================
# MAP
# =========================
st.subheader("🗺️ India Pollution Map")
city_avg = df.groupby('City')['AQI'].mean().reset_index()
coords = {"Delhi": (28.61,77.20),"Mumbai": (19.07,72.87),"Chennai": (13.08,80.27),
          "Bengaluru": (12.97,77.59),"Kolkata": (22.57,88.36)}
city_avg['lat'] = city_avg['City'].map(lambda x: coords.get(x,(20,77))[0])
city_avg['lon'] = city_avg['City'].map(lambda x: coords.get(x,(20,77))[1])
fig_map = px.scatter_geo(city_avg, lat='lat', lon='lon', color='AQI',
                        size='AQI', color_continuous_scale='Reds')
st.plotly_chart(fig_map, use_container_width=True)

# =========================
# COMPARISON
# =========================
compare = st.multiselect("Compare Cities", all_cities, default=[sel_city])
fig_c = px.line(df[df['City'].isin(compare)], x='Datetime', y='AQI', color='City')
st.plotly_chart(fig_c, use_container_width=True)

# =========================
# NEW: CROSS-DATASET FUSION ANALYSIS
# =========================
st.divider()
st.subheader("🔗 Macro-Micro Fusion Analysis")
st.write("Comparing Local Air Quality (Micro) with National Carbon Output (Macro)")

# Link City to State (Approximation)
city_state_map = {"Delhi": "Delhi", "Mumbai": "Maharashtra", "Ahmedabad": "Gujarat", 
                  "Bengaluru": "Karnataka", "Chennai": "Tamil Nadu", "Hyderabad": "Telangana"}
sel_state_mapped = city_state_map.get(sel_city, "Delhi")

# Calculate Fusion Metric: Pollution Efficiency
# MtCO2 per AQI unit (Yearly)
city_yearly = state_df.groupby('Year')['AQI'].mean().reset_index()
state_yearly = co2_df[co2_df['State'] == sel_state_mapped].sort_values('Year')
fusion_df = pd.merge(city_yearly, state_yearly, on='Year')

if not fusion_df.empty:
    fig_fusion = go.Figure()
    fig_fusion.add_trace(go.Scatter(x=fusion_df['Year'], y=fusion_df['AQI'], name="City AQI", yaxis="y1"))
    fig_fusion.add_trace(go.Bar(x=fusion_df['Year'], y=fusion_df['Carbon_Emissions_MtCO2'], name="State CO2 (MtCO2)", yaxis="y2", opacity=0.3))
    
    fig_fusion.update_layout(
        title=f"AQI vs CO2 Correlation: {sel_city} ({sel_state_mapped})",
        yaxis=dict(title="AQI Score"),
        yaxis2=dict(title="MtCO2 Emissions", overlaying="y", side="right"),
        template="plotly_dark"
    )
    st.plotly_chart(fig_fusion, use_container_width=True)
else:
    st.info("Insufficient overlapping years between datasets for fusion analysis.")

# =========================
# CO2 ANALYSIS
# =========================
st.subheader("🌍 Environmental Indicators")
fig_env = px.scatter(co2_df, x='GDP_BillionINR', y='Carbon_Emissions_MtCO2',
                      size='Energy_Use_TWh', color='Urbanization_Percent',
                      hover_name='State', title="Economic Growth vs Emissions")
st.plotly_chart(fig_env, use_container_width=True)

# =========================
# CORRELATION
# =========================
st.subheader("📊 Correlation Matrix")
corr = co2_df[['Carbon_Emissions_MtCO2','GDP_BillionINR','Urbanization_Percent','Energy_Use_TWh']].corr()
st.plotly_chart(px.imshow(corr, text_auto=True, title="Industrial Factor Correlation"), use_container_width=True)

# =========================
# SUSTAINABILITY SCORE
# =========================
st.subheader("🌱 Sustainability Score")
states = sorted(co2_df['State'].unique())
sel_state = st.selectbox("State", states, index=states.index(sel_state_mapped) if sel_state_mapped in states else 0)

scaler = MinMaxScaler()
norm_cols = ['Carbon_Emissions_MtCO2','GDP_BillionINR','Urbanization_Percent','Energy_Use_TWh']
norm = scaler.fit_transform(co2_df[norm_cols])
co2_df[['co2_n','gdp_n','urb_n','energy_n']] = norm

row = co2_df[co2_df['State'] == sel_state].tail(1)
score = (0.4*(1-row['co2_n'].values[0]) + 0.2*row['gdp_n'].values[0] +
         0.2*(1-row['energy_n'].values[0]) + 0.2*(1-row['urb_n'].values[0]))
st.metric(f"Sustainability Score: {sel_state}", f"{score:.2f}")

# =========================
# 🔮 FORECAST 1: AQI (UNCHANGED)
# =========================
st.divider()
st.subheader("🔮 City AQI Predictive Engine")

f_state = st.selectbox("Forecast Target", all_cities, index=list(all_cities).index(sel_city), key="aqi_target")
f_freq = st.radio("Interval", ["Daily","Weekly","Monthly","Yearly"], key="aqi_freq")
f_steps = st.number_input("Steps", 1, 30, 7, key="aqi_steps")

if st.button("Generate AQI Forecast", key="aqi_btn"):
    try:
        with st.spinner("🧠 Training AQI AI..."):
            freq_map = {"Daily": 'D', "Weekly": 'W', "Monthly": 'M', "Yearly": 'Y'}
            p_data = df[df['City']==f_state].set_index('Datetime')['AQI'].resample(freq_map[f_freq]).mean().ffill()

            window = 14 if f_freq!="Yearly" else 2
            train_df = pd.DataFrame({'AQI': p_data.values})
            train_df['M_Sin'] = np.sin(2*np.pi*p_data.index.month/12)
            train_df['M_Cos'] = np.cos(2*np.pi*p_data.index.month/12)
            train_df['Trend'] = np.arange(len(train_df))/len(train_df)

            pt = PowerTransformer(method='yeo-johnson')
            scaled = pt.fit_transform(train_df)

            X,y=[],[]
            for i in range(len(scaled)-window):
                X.append(scaled[i:i+window])
                y.append(scaled[i+window,0])

            model = Sequential([
                LSTM(32, return_sequences=True, input_shape=(window,4)),
                Dropout(0.2),
                LSTM(16),
                Dense(1)
            ])

            model.compile(optimizer='adam', loss='mse')
            model.fit(np.array(X), np.array(y), epochs=60, verbose=0)

            last_w = scaled[-window:].reshape(1,window,4)
            preds=[]
            curr_date=p_data.index[-1]

            for i in range(f_steps):
                p=model.predict(last_w,verbose=0)[0,0]
                preds.append(p)
                curr_date += pd.DateOffset(months=1)
                new_row=np.array([p,np.sin(2*np.pi*curr_date.month/12),
                                  np.cos(2*np.pi*curr_date.month/12),
                                  (len(p_data)+i)/len(p_data)]).reshape(1,1,4)
                last_w=np.append(last_w[:,1:,:],new_row,axis=1)

            res_dummy=np.zeros((len(preds),4))
            res_dummy[:,0]=preds
            res=pt.inverse_transform(res_dummy)[:,0]
            f_dates=pd.date_range(p_data.index[-1],periods=f_steps+1,freq=freq_map[f_freq])[1:]

            fig_f=go.Figure()
            fig_f.add_trace(go.Scatter(x=p_data.index[-60:],y=p_data.values[-60:],name="Past"))
            fig_f.add_trace(go.Scatter(x=f_dates,y=res,name="Forecast",line=dict(dash='dash')))
            st.plotly_chart(fig_f, use_container_width=True)
            st.table(pd.DataFrame({"Date":f_dates,"AQI":res.astype(int)}))
    except Exception as e:
        st.error(e)

# =========================
# NEW 🔮 FORECAST 2: CARBON EMISSIONS (Using your logic)
# =========================
st.divider()
st.subheader("🔮 Industrial CO2 Predictive Engine")
st.write("Using AI to forecast state-level carbon footprints based on historic energy and urbanization trends.")

f_state_co2 = st.selectbox("Select State for CO2 Forecast", states, key="co2_target")
f_steps_co2 = st.number_input("Forecast Years", 1, 10, 5, key="co2_steps")

if st.button("Generate CO2 Forecast", key="co2_btn"):
    try:
        with st.spinner("🧠 Training CO2 AI..."):
            # Prepare Yearly Data
            p_data_co2 = co2_df[co2_df['State'] == f_state_co2].sort_values('Year').set_index('Year')['Carbon_Emissions_MtCO2']
            
            # Using your window=2 logic for Yearly data
            window_co2 = 2
            train_df_co2 = pd.DataFrame({'CO2': p_data_co2.values})
            # Trend feature exactly like your logic
            train_df_co2['Trend'] = np.arange(len(train_df_co2))/len(train_df_co2)
            # Dummy seasonality for annual (since month isn't applicable, we use 0)
            train_df_co2['M_Sin'] = 0
            train_df_co2['M_Cos'] = 0

            pt_co2 = PowerTransformer(method='yeo-johnson')
            scaled_co2 = pt_co2.fit_transform(train_df_co2)

            X_c, y_c = [], []
            for i in range(len(scaled_co2)-window_co2):
                X_c.append(scaled_co2[i:i+window_co2])
                y_c.append(scaled_co2[i+window_co2, 0])

            model_co2 = Sequential([
                LSTM(32, return_sequences=True, input_shape=(window_co2, 4)),
                Dropout(0.2),
                LSTM(16),
                Dense(1)
            ])

            model_co2.compile(optimizer='adam', loss='mse')
            model_co2.fit(np.array(X_c), np.array(y_c), epochs=100, verbose=0)

            last_w_c = scaled_co2[-window_co2:].reshape(1,window_co2,4)
            preds_c = []
            for i in range(f_steps_co2):
                p = model_co2.predict(last_w_c, verbose=0)[0,0]
                preds_c.append(p)
                new_row_c = np.array([p, 0, 0, (len(p_data_co2)+i)/len(p_data_co2)]).reshape(1,1,4)
                last_w_c = np.append(last_w_c[:,1:,:], new_row_c, axis=1)

            res_dummy_c = np.zeros((len(preds_c), 4))
            res_dummy_c[:,0] = preds_c
            res_c = pt_co2.inverse_transform(res_dummy_c)[:,0]
            
            f_years = np.arange(p_data_co2.index[-1]+1, p_data_co2.index[-1]+1+f_steps_co2)

            fig_f_c = go.Figure()
            fig_f_c.add_trace(go.Scatter(x=p_data_co2.index, y=p_data_co2.values, name="Historical Emissions"))
            fig_f_c.add_trace(go.Scatter(x=f_years, y=res_c, name="AI Forecast", line=dict(dash='dash', color='red')))
            fig_f_c.update_layout(title=f"CO2 Forecast for {f_state_co2} (MtCO2)", template="plotly_dark")
            st.plotly_chart(fig_f_c, use_container_width=True)
            st.table(pd.DataFrame({"Year": f_years, "Estimated MtCO2": res_c.astype(float).round(2)}))

    except Exception as e:
        st.error(f"CO2 Prediction Error: {e}")

# =========================
# COMMUNITY ACTION
# =========================
st.subheader("🌱 Community Action")
st.markdown("""
- Use public transport 🚶  
- Avoid burning waste 🔥  
- Plant trees 🌳  
- Reduce electricity usage ⚡  
""")

# =========================
# AI EXPLANATION
# =========================
st.subheader("🤖 AI Explanation")
st.info("The system uses a Multi-layer LSTM (Long Short-Term Memory) network. For AQI, it processes seasonal variables. For Carbon, it processes long-term economic trends.")

# =========================
# FOOTER
# =========================
st.sidebar.write("👥 Team: Data | ML | Frontend | Research")