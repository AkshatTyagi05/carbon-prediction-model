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

st.set_page_config(page_title="Carbon Emission Analysis and Forecasting", layout="wide")

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
st.title("🌍 Carbon Emission Analysis and Forecasting")
st.caption("CEAF v2.0 | AI-powered Environmental Analysis & Decision Support System")


# =========================
# CO2 RANKING (REPLACES KPI)
# =========================
st.subheader("🏆 State Carbon Emission Ranking")

latest_co2_rank = co2_df.sort_values('Year').groupby('State').tail(1)
ranking = latest_co2_rank.sort_values('Carbon_Emissions_MtCO2')

c1, c2, c3 = st.columns(3)
c1.metric("🌿 Cleanest State", ranking.iloc[0]['State'])
c2.metric("🏭 Most Polluting State", ranking.iloc[-1]['State'])
c3.metric("📊 Avg CO2", f"{latest_co2_rank['Carbon_Emissions_MtCO2'].mean():.2f}")

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
# =========================
# MAP (NOW CO2 BASED)
# =========================
st.subheader("🗺️ India Carbon Emission Map")

# Use latest CO2 value per state
latest_co2 = co2_df.sort_values('Year').groupby('State').tail(1)

coords = {
    "Delhi": (28.61,77.20), "Maharashtra": (19.75,75.71),
    "Tamil Nadu": (11.12,78.65), "Karnataka": (15.31,75.71),
    "West Bengal": (22.98,87.85), "Gujarat": (22.25,71.19),
    "Telangana": (18.11,79.01), "Uttar Pradesh": (26.85,80.95),
    "Rajasthan": (27.02,74.21), "Madhya Pradesh": (23.47,77.95),
    "Punjab": (31.14,75.34), "Haryana": (29.06,76.08),
    "Bihar": (25.09,85.31), "Odisha": (20.29,85.82),
    "Kerala": (10.85,76.27), "Assam": (26.20,92.93)
}

latest_co2['lat'] = latest_co2['State'].map(lambda x: coords.get(x,(20,77))[0])
latest_co2['lon'] = latest_co2['State'].map(lambda x: coords.get(x,(20,77))[1])

fig_map = px.scatter_geo(
    latest_co2,
    lat='lat',
    lon='lon',
    color='Carbon_Emissions_MtCO2',
    size='Carbon_Emissions_MtCO2',
    hover_name='State',
    color_continuous_scale='Reds',
    title="State-wise Carbon Emissions"
)

st.plotly_chart(fig_map, use_container_width=True)

# =========================
# COMPARISON (CLEAN)
# =========================
st.subheader("📊 City Comparison (Smoothed)")

compare = st.multiselect("Compare Cities", all_cities, default=[sel_city])

if len(compare) > 0:
    comp_df = df[df['City'].isin(compare)].copy()

    comp_df = comp_df.set_index('Datetime').groupby('City')['AQI'] \
                     .resample('M').mean().reset_index()

    comp_df['AQI'] = comp_df.groupby('City')['AQI'] \
                           .transform(lambda x: x.rolling(3, min_periods=1).mean())

    fig_c = px.line(comp_df, x='Datetime', y='AQI', color='City',
                    title="Monthly Smoothed AQI Comparison",
                    template="plotly_dark")

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
            model.fit(np.array(X), np.array(y), epochs=35, verbose=0)

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
# =========================
# NEW 🔮 FORECAST 2: CARBON EMISSIONS (SAME LOGIC AS AQI)
# =========================
st.divider()
st.subheader("🔮 Industrial CO2 Predictive Engine")

f_state_co2 = st.selectbox("Select State for CO2 Forecast", states, key="co2_target")
f_freq_co2 = st.radio("Interval", ["Yearly"], key="co2_freq")  # Only yearly valid
f_steps_co2 = st.number_input("Steps", 1, 10, 5, key="co2_steps")

if st.button("Generate CO2 Forecast", key="co2_btn"):
    try:
        with st.spinner("🧠 Training CO2 AI..."):

            # EXACT SAME FLOW AS AQI
            p_data = co2_df[co2_df['State']==f_state_co2] \
                        .sort_values('Year') \
                        .set_index('Year')['Carbon_Emissions_MtCO2']

            window = 2  # same as yearly AQI

            train_df = pd.DataFrame({'AQI': p_data.values})  # keep same naming for logic consistency
            train_df['M_Sin'] = np.sin(2*np.pi*np.arange(len(p_data))/len(p_data))
            train_df['M_Cos'] = np.cos(2*np.pi*np.arange(len(p_data))/len(p_data))
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
            curr_year = p_data.index[-1]

            for i in range(f_steps_co2):
                p=model.predict(last_w,verbose=0)[0,0]
                preds.append(p)

                curr_year += 1

                new_row=np.array([
                    p,
                    np.sin(2*np.pi*(len(p_data)+i)/len(p_data)),
                    np.cos(2*np.pi*(len(p_data)+i)/len(p_data)),
                    (len(p_data)+i)/len(p_data)
                ]).reshape(1,1,4)

                last_w=np.append(last_w[:,1:,:],new_row,axis=1)

            res_dummy=np.zeros((len(preds),4))
            res_dummy[:,0]=preds
            res=pt.inverse_transform(res_dummy)[:,0]

            f_years = np.arange(p_data.index[-1]+1, p_data.index[-1]+1+f_steps_co2)

            fig_f=go.Figure()
            fig_f.add_trace(go.Scatter(x=p_data.index,y=p_data.values,name="Past"))
            fig_f.add_trace(go.Scatter(x=f_years,y=res,name="Forecast",
                                      line=dict(dash='dash',color='red')))
            fig_f.update_layout(title=f"CO2 Forecast: {f_state_co2}", template="plotly_dark")

            st.plotly_chart(fig_f, use_container_width=True)

            st.table(pd.DataFrame({
                "Year": f_years,
                "Forecasted CO2 (MtCO2)": res.round(2)
            }))

    except Exception as e:
        st.error(f"CO2 Prediction Error: {e}")


# =========================
# ⚡ ENERGY IMPACT SIMULATOR
# =========================
st.subheader("⚡ Energy Impact Simulator")

energy_reduction = st.slider("Reduce Energy Use (%)", 0, 50, 10)

impact = row['Carbon_Emissions_MtCO2'].values[0] * (1 - energy_reduction/100)

st.metric("Estimated CO2 After Reduction", f"{impact:.2f} MtCO2")


# =========================
# COMMUNITY ACTION
# =========================
st.subheader("Community Action 🌱 ")
st.markdown("""
- Use public transport 🚶  
- Avoid burning waste 🔥  
- Plant trees 🌳  
- Reduce electricity usage ⚡  
""")

# =========================
# 📢 COMMUNITY REPORT SYSTEM
# =========================
st.divider()
st.subheader("📢 Report Local Pollution")

report_text = st.text_area("Describe the issue")
location = st.text_input("Location")

if st.button("Submit Report"):
    if report_text.strip():
        new_report = pd.DataFrame({
            "Issue": [report_text],
            "Location": [location],
            "Time": [pd.Timestamp.now()]
        })

        try:
            old = pd.read_csv("reports.csv")
            new_report = pd.concat([old, new_report], ignore_index=True)
        except:
            pass

        new_report.to_csv("reports.csv", index=False)
        st.success("✅ Report saved successfully!")
    else:
        st.warning("Please enter issue.")

# =========================
# AI EXPLANATION
# =========================
st.subheader("🤖 AI Explanation")
st.info("The system uses a Multi-layer LSTM (Long Short-Term Memory) network. For AQI, it processes seasonal variables. For Carbon, it processes long-term economic trends.")

# =========================
# FOOTER
# =========================
st.sidebar.write("👥 Team: Data | ML | Frontend | Research")

