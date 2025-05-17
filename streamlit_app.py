# 📈 주가 예측 모델 비교 실습 (XGBoost vs LSTM + Streamlit 대시보드)

# ✅ 1. yfinance로 데이터 수집
!pip install yfinance xgboost streamlit
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

stock = 'AAPL'
df = yf.download(stock, start='2020-01-01', end='2024-12-31')
df = df[['Close']].dropna()
df['return'] = df['Close'].pct_change().fillna(0)

# ✅ 2. XGBoost 모델 (회귀 기반)
from xgboost import XGBRegressor

df['lag1'] = df['Close'].shift(1)
df['lag2'] = df['Close'].shift(2)
df = df.dropna()

X = df[['lag1', 'lag2']]
y = df['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

# ✅ 3. LSTM 모델 (시계열 기반)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Close']])

sequence_length = 10
X_lstm, y_lstm = [], []
for i in range(len(scaled_data) - sequence_length):
    X_lstm.append(scaled_data[i:i+sequence_length])
    y_lstm.append(scaled_data[i+sequence_length])
X_lstm = np.arry(y_lstm)

split = int(len(X_ay(X_lstm)
y_lstm = np.arralstm) * 0.8)
X_train_lstm, X_test_lstm = X_lstm[:split], X_lstm[split:]
y_train_lstm, y_test_lstm = y_lstm[:split], y_lstm[split:]

model = Sequential([
    Input(shape=(sequence_length, 1)),
    LSTM(32),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train_lstm, y_train_lstm, epochs=10, verbose=0)

y_pred_lstm = model.predict(X_test_lstm)
y_pred_lstm_rescaled = scaler.inverse_transform(y_pred_lstm)
y_test_lstm_rescaled = scaler.inverse_transform(y_test_lstm)

rmse_lstm = np.sqrt(mean_squared_error(y_test_lstm_rescaled, y_pred_lstm_rescaled))

# ✅ 4. Streamlit 대시보드 구성
import streamlit as st
import matplotlib.pyplot as plt

st.title("📈 주가 예측 모델 비교 대시보드")
st.markdown(f"**분석 종목:** {stock}")
st.markdown(f"**XGBoost RMSE:** {rmse_xgb:.2f}")
st.markdown(f"**LSTM RMSE:** {rmse_lstm:.2f}")

st.subheader("XGBoost 예측 vs 실제")
fig1, ax1 = plt.subplots(figsize=(12,5))
ax1.plot(y_test.values[:100], label='실제')
ax1.plot(y_pred_xgb[:100], label='예측')
ax1.legend()
st.pyplot(fig1)

st.subheader("LSTM 예측 vs 실제")
fig2, ax2 = plt.subplots(figsize=(12,5))
ax2.plot(y_test_lstm_rescaled[:100], label='실제')
ax2.plot(y_pred_lstm_rescaled[:100], label='예측')
ax2.legend()
st.pyplot(fig2)

st.markdown("---")
st.write("이 대시보드는 XGBoost와 LSTM 모델의 주가 예측 결과를 비교합니다. 정확도(RMSE)가 낮을수록 더 좋은 예측입니다.")
