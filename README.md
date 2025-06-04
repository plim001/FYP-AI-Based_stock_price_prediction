# FYP-AI-Based_stock_price_prediction
Predicting Stock Prices Using LSTM, RNN, and XGBoost with Twitter Sentiment Analysis — A comprehensive machine learning project for time-series forecasting that combines deep learning, gradient boosting, and social media insights.

# Stock Price Prediction for BYD (BYDDY), Li-Auto (LI), NIO (NIO), XPENG (XPEV), and TESLA (TSLA)

This repository contains a comprehensive project that explores multiple machine learning models to predict stock prices for BYD Company Limited (BYDDY). The models implemented include LSTM (Long Short-Term Memory), RNN (Recurrent Neural Network), and XGBoost. Additionally, the project incorporates sentiment analysis from Twitter data to assess the correlation between public sentiment and stock performance.

## 📁 Project Structure (repeat for other selected stocks)

- `BYD's LSTM Model.ipynb` – Implements a deep learning approach using LSTM for time-series prediction of BYDDY closing prices.
- `BYD's RNN Model.ipynb` – Utilizes a stacked SimpleRNN architecture to model the temporal dependencies in the stock data.
- `BYD's XGBoost Model.ipynb` – Leverages gradient boosting (XGBoost) with extensive feature engineering including lag features and rolling statistics.
- `Data Cleaning.ipynb` – Cleans raw Twitter data for companies like Li Auto, XPeng, and NIO for further analysis.
- `Data Wangling (Twitter).ipynb` – Collects and processes tweets using `snscrape`, preparing datasets for sentiment and textual analysis.

## 📊 Datasets (repeat for other selected stocks)

- Stock price data: Acquired using the `yfinance` API for ticker `BYDDY` between 2020-07-30 and 2022-08-27.
- Twitter data: Scraped using `snscrape` and includes tweets mentioning $LI, $NIO, and $XPEV.

## 🔍 Models Overview

### 🔹 LSTM
- Framework: TensorFlow/Keras
- Input: 60-day window of past prices
- Output: Next day's closing price
- Data scaling: `MinMaxScaler`

### 🔹 RNN
- Framework: Keras
- Sequence length: 20 days
- Layers: 4 stacked SimpleRNN layers with dropout
- Loss: Mean Squared Error (MSE)

### 🔹 XGBoost
- Features:
  - Lagged prices, volume, price ranges
  - Rolling mean and standard deviation
- Evaluation:
  - RMSE
  - MAPE
- Tools: `XGBRegressor` from `xgboost`, `StandardScaler`

## 🧹 Data Preprocessing

- Duplicate removal
- Text cleaning: Removing mentions, hashtags, URLs, punctuation, and stopwords
- Time formatting and splitting into `Date2` and `Time` columns
- CSV export for model consumption

## 📈 Evaluation Metrics

- **LSTM / RNN**:
  - Loss: Mean Squared Error
  - Metric: Training accuracy and visual comparison with actual prices

- **XGBoost**:
  - RMSE (Root Mean Squared Error)
  - MAPE (Mean Absolute Percentage Error)

## 📌 Requirements

```bash
pip install pandas numpy matplotlib yfinance seaborn scikit-learn xgboost keras tensorflow snscrape
