#  Stock Price Prediction with BiLSTM Model

A comprehensive **stock market analysis and portfolio management platform** powered by **Bidirectional LSTM (BiLSTM)** deep learning models. This project combines advanced time series forecasting with real-time market data to provide intelligent stock predictions, portfolio tracking, and technical analysis.
---

##  Project Overview

This project demonstrates an **end-to-end machine learning pipeline** for stock market prediction and portfolio management:

- **Train BiLSTM models** on historical stock data  for 10 major tech stocks
- **Real-time predictions** with confidence scoring and trading signals
- **Interactive dashboard** for portfolio tracking, technical analysis, and performance visualization
- **Multi-stock analysis** with sector allocation and risk metrics
- **Export capabilities** for reports and portfolio data

The system achieves impressive results with **R² scores ranging from 0.89 to 0.99** across different stocks, demonstrating strong predictive capabilities for short to medium-term price movements.

---

##  Features

###  **Machine Learning Models**
- **Bidirectional LSTM architecture** for capturing temporal patterns in both directions
- **Dual-layer BiLSTM** with dropout regularization 
- **Sequence length**: 60 days of historical data
- **Early stopping** and learning rate reduction for optimal convergence
- Models trained on **10 major tech stocks**: AAPL, GOOGL, MSFT, AMZN, TSLA, NVDA, META, NFLX, AMD, INTC

###  **Interactive Dashboard**
- **5 Main Pages**:
  -  **Dashboard**: Portfolio overview, market indices, top performers
  -  **Stock Predictions**: Multi-day price forecasting with confidence metrics
  -  **Portfolio Tracker**: Real-time P&L tracking, sector allocation
  -  **Technical Analysis**: RSI, MACD, Bollinger Bands, volume analysis
  -  **Reports & Export**: CSV export, performance reports, dividend tracking

###  **Technical Analysis**
- **Moving Averages** (MA 20, 50, 200)
- **RSI** (Relative Strength Index) with overbought/oversold signals
- **MACD** (Moving Average Convergence Divergence)
- **Bollinger Bands** for volatility assessment
- **Volume analysis** with moving averages

###  **Portfolio Management**
- Add/remove stocks with buy price and date tracking
- Real-time P&L calculation (absolute & percentage)
- **Risk metrics**: Portfolio volatility, Sharpe ratio, VaR (95%)
- Sector allocation visualization
- Dividend tracking for income-generating stocks


---

##  Model Performance

### **Training Results Summary**

| Ticker | Train RMSE | Test RMSE | Test MAE | Test R²  |
|--------|-----------|-----------|----------|----------|
| AAPL   | $3.10     | $6.80     | $5.48    | **0.9193** |
| GOOGL  | $2.38     | $4.21     | $3.26    | **0.9573** |
| MSFT   | $5.18     | $9.12     | $7.50    | **0.9535** |
| AMZN   | $3.44     | $4.29     | $3.33    | **0.9756** |
| TSLA   | $8.43     | $12.47    | $8.98    | **0.9548** |
| NVDA   | $0.73     | $4.49     | $3.12    | **0.9853** |
| META   | $7.59     | $14.60    | $11.06   | **0.9818** |
| NFLX   | $13.77    | $16.30    | $11.87   | **0.9877**  |
| AMD    | $3.48     | $8.47     | $6.59    | **0.8989** |
| INTC   | $1.13     | $1.28     | $0.91    | **0.9763** |

**Average R²**: 0.9525 across all models

---


##  Installation & Setup

### **1. Clone Repository**
```bash
git clone https://github.com/Zeyad-Baloch/stock-price-predictor.git
cd stock-price-predictor
```

### **2. Create Virtual Environment** (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**requirements.txt**:
```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
plotly>=5.0.0
yfinance>=0.1.70
tensorflow>=2.10.0
scikit-learn>=1.0.0
streamlit>=1.20.0
```

### **4. Train Models** (Optional - Pre-trained models included)
```bash
python train_models.py
```
This will:
- Download 8 years of historical data (2017-2025)
- Train BiLSTM models for 10 stocks
- Save models to `trained_stock_models/`
- Generate performance visualizations

### **5. Launch Dashboard**
```bash
streamlit run app.py
```


---

##  Usage Guide

### **Making Predictions**
1. Navigate to **Stock Predictions** page
2. Select a ticker from the dropdown
3. Choose prediction horizon (1-30 days)
4. Click **Generate Predictions**
5. View price forecast, confidence metrics, and trading signals

### **Managing Portfolio**
1. Go to **Portfolio Tracker** page
2. Add stocks with ticker, shares, buy price, and date
3. Click **Refresh Prices** to update current values
4. Monitor P&L, sector allocation, and risk metrics
5. Export data via ** Reports & Export** page

### **Technical Analysis**
1. Open **Technical Analysis** page
2. Enter stock symbol and select time period
3. View candlestick charts with indicators
4. Analyze RSI, MACD, Bollinger Bands, and volume


---

##  Performance Insights

### **Why BiLSTM Outperforms Standard LSTM**
- **Bidirectional context**: Learns from both past and future patterns
- **Better gradient flow**: Reduces vanishing gradient problems
- **Temporal dependencies**: Captures long-term trends effectively

### **Model Strengths**
-  Excellent R² scores (0.89-0.99) indicate strong predictive power
-  Low RMSE relative to stock prices shows accurate forecasting
-  Stable across different volatility regimes (TSLA vs INTC)

### **Limitations**
-  **Black swan events**: Models can't predict unprecedented market shocks
-  **Overfitting risk**: High R² on historical data doesn't guarantee future performance
-  **Short-term focus**: Best for 1-30 day predictions, not long-term investing

---

##  Future Enhancements

### **Model Improvements**
- [ ] **Attention mechanisms** for dynamic feature weighting
- [ ] **Multi-variate models** incorporating volume, sentiment, macroeconomic data
- [ ] **Ensemble methods** combining BiLSTM with GRU, Transformer models
- [ ] **Transfer learning** from larger stock universes

### **Feature Additions**
- [ ] **News sentiment analysis** using NLP on financial headlines
- [ ] **Options pricing** predictions for derivatives trading
- [ ] **Backtesting engine** for strategy validation
- [ ] **Real-time alerts** via email/SMS for signal triggers


### **Technical Enhancements**
- [ ] **Docker containerization** for easy deployment
- [ ] **Cloud deployment** on AWS/GCP/Azure
- [ ] **API endpoints** for programmatic access
- [ ] **A/B testing framework** for model comparison

---


##  Contributing

Contributions are welcome! Here's how you can help:


### **Areas for Contribution**
-  Bug fixes and performance optimizations
-  Additional technical indicators (Fibonacci, Ichimoku, etc.)
-  UI/UX improvements for dashboard
-  Documentation and tutorials
-  Unit tests and integration tests

---
