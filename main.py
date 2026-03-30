import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ydata_profiling import ProfileReport
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense

# --- 1. CONFIGURATION ---
DATA_PATH = r"C:\Private\epics2\city_day.csv" # Update this path!
OUTPUT_DIR = Path("./delhi_aqi_final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # --- 2. LOAD & CLEAN DATA ---
    print("Loading Data...")
    df = pd.read_csv(DATA_PATH)
    
    # 1. Filter for Delhi 
    # 2. Use 'Datetime' as the time column
    # 3. Use 'CO' as the carbon emission proxy
    delhi = df[df['City'] == 'Delhi'].copy()
    delhi['Datetime'] = pd.to_datetime(delhi['Datetime'])
    delhi = delhi.sort_values('Datetime').dropna(subset=['CO'])
    
    # Resample to Daily Mean to make the AI training more stable
    delhi_daily = delhi.set_index('Datetime')['CO'].resample('D').mean().fillna(method='ffill')
    final_df = delhi_daily.reset_index()
    final_df.columns = ['Date', 'CO_Level']

    # --- 3. THE BEAUTIFUL EDA REPORT ---
    print("Generating Professional HTML EDA Report...")
    # Removing 'dark_mode' to avoid previous error
    profile = ProfileReport(final_df, title="Delhi Carbon Monoxide Analysis", explorative=True)
    profile.to_file(OUTPUT_DIR / "Delhi_EDA_Report.html")

    # --- 4. PREPARE AI DATA ---
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(final_df[['CO_Level']])

    # Use last 14 days to predict the next 1 day
    window = 14
    X, y = [], []
    for i in range(len(scaled_data) - window):
        X.append(scaled_data[i:i+window])
        y.append(scaled_data[i+window])
    X, y = np.array(X), np.array(y)

    # Split 80/20
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # --- 5. TRAIN MODELS (LSTM, GRU, RNN) ---
    def train_model(m_type):
        print(f"Training {m_type} Model...")
        model = Sequential()
        if m_type == "LSTM":
            model.add(LSTM(50, activation='relu', input_shape=(window, 1)))
        elif m_type == "GRU":
            model.add(GRU(50, activation='relu', input_shape=(window, 1)))
        else:
            model.add(SimpleRNN(50, activation='relu', input_shape=(window, 1)))
        
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
        return model

    m_lstm = train_model("LSTM")
    m_gru = train_model("GRU")
    m_rnn = train_model("RNN")

    # --- 6. FUTURE FORECAST (Next 30 Days) ---
    print("Predicting the next 30 days...")
    last_window = scaled_data[-window:].reshape(1, window, 1)
    future_preds = []
    for _ in range(30):
        pred = m_lstm.predict(last_window, verbose=0)[0,0]
        future_preds.append(pred)
        last_window = np.append(last_window[:, 1:, :], [[[pred]]], axis=1)
    
    final_future = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

    # --- 7. FINAL VISUALS ---
    plt.figure(figsize=(12, 6))
    plt.plot(final_df['Date'].iloc[-60:], final_df['CO_Level'].iloc[-60:], label="Past Data", color='black')
    
    future_dates = pd.date_range(start=final_df['Date'].iloc[-1], periods=31)[1:]
    plt.plot(future_dates, final_future, label="AI Future Forecast", color='red', linestyle='--', marker='o')
    
    plt.title("Delhi Carbon Monoxide (CO) Forecast: Next 30 Days")
    plt.xlabel("Date")
    plt.ylabel("CO Level")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "Delhi_Future_Forecast.png")
    
    print(f"\nSUCCESS! Results generated in {OUTPUT_DIR}")
    print("1. Delhi_EDA_Report.html")
    print("2. Delhi_Future_Forecast.png")

if __name__ == "__main__":
    main()