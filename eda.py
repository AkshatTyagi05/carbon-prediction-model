import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense
import matplotlib.pyplot as plt
import seaborn as sns
import ydata_profiling as yp  # pip install ydata-profiling

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("CO2 emissions India.csv")

# =========================
# CLEAN DATA
# =========================
df = df.sort_values(['State', 'Year'])

# Fill missing values
df = df.fillna(method='ffill').fillna(method='bfill')

# Save CLEAN DATA
df.to_csv("clean_co2_data.csv", index=False)

print("✅ Clean data saved: clean_co2_data.csv")

# =========================
# NORMALIZATION
# =========================
features = [
    'Carbon_Emissions_MtCO2',
    'GDP_BillionINR',
    'Urbanization_Percent',
    'Energy_Use_TWh'
]

scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[features] = scaler.fit_transform(df[features])

# Save NORMALIZED DATA
df_scaled.to_csv("normalized_co2_data.csv", index=False)

print("✅ Normalized data saved: normalized_co2_data.csv")

# =========================
# EDA REPORT (HTML)
# =========================
profile = yp.ProfileReport(df, title="CO2 Emissions EDA Report", explorative=True)
profile.to_file("co2_eda_report.html")

print("📊 EDA Report generated: co2_eda_report.html")

# =========================
# MODEL PREPARATION
# =========================
# Use one state for modeling (example: Maharashtra)
state = df_scaled['State'].unique()[0]
state_df = df_scaled[df_scaled['State'] == state]

series = state_df['Carbon_Emissions_MtCO2'].values

# Windowing
window = 3
X, y = [], []

for i in range(len(series) - window):
    X.append(series[i:i+window])
    y.append(series[i+window])

X = np.array(X)
y = np.array(y)

# Reshape for RNN models
X = X.reshape((X.shape[0], X.shape[1], 1))

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =========================
# MODEL FUNCTIONS
# =========================
def build_lstm():
    model = Sequential([
        LSTM(32, input_shape=(window,1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_gru():
    model = Sequential([
        GRU(32, input_shape=(window,1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_rnn():
    model = Sequential([
        SimpleRNN(32, input_shape=(window,1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# =========================
# TRAIN MODELS
# =========================
models = {
    "LSTM": build_lstm(),
    "GRU": build_gru(),
    "RNN": build_rnn()
}

history = {}
predictions = {}

for name, model in models.items():
    print(f"🚀 Training {name}...")
    
    h = model.fit(X_train, y_train, epochs=50, verbose=0)
    history[name] = h.history['loss']
    
    preds = model.predict(X_test)
    predictions[name] = preds.flatten()

# =========================
# PERFORMANCE COMPARISON
# =========================
plt.figure(figsize=(10,5))

for name in history:
    plt.plot(history[name], label=name)

plt.title("Training Loss Comparison")
plt.legend()
plt.savefig("model_loss_comparison.png")
plt.show()

print("📈 Saved: model_loss_comparison.png")

# =========================
# PREDICTION COMPARISON
# =========================
plt.figure(figsize=(10,5))

plt.plot(y_test, label="Actual", linewidth=2)

for name in predictions:
    plt.plot(predictions[name], label=name)

plt.title(f"Model Predictions ({state})")
plt.legend()
plt.savefig("model_prediction_comparison.png")
plt.show()

print("📊 Saved: model_prediction_comparison.png")