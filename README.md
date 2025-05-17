# 📈 Stock Price Prediction Dashboard

This Streamlit app compares two machine learning models (XGBoost and LSTM) to predict stock prices using historical data.

## 🚀 Features
- Downloads stock data using `yfinance`
- Predicts closing price with both XGBoost and LSTM models
- Visualizes actual vs predicted prices
- Compares model performance with RMSE

## 📦 Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

## ▶️ Run locally
```bash
streamlit run app.py
```

## ☁️ Deploy on Streamlit Cloud
1. Fork this repository
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Click "New app", connect your GitHub, and select `app.py`
4. Click Deploy 🎉

## 🧠 Models used
- XGBoost: gradient-boosted tree for tabular prediction
- LSTM: deep learning model for time series prediction
