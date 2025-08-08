import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta


class StockPredictor:
    def __init__(self, model_loader, data_fetcher):
        self.model_loader = model_loader
        self.data_fetcher = data_fetcher

    def prepare_prediction_data(self, data, sequence_length):

        close_prices = data['Close'].values.reshape(-1, 1)


        if len(close_prices) < sequence_length:
            st.error(f"Not enough data. Need at least {sequence_length} days.")
            return None

        last_sequence = close_prices[-sequence_length:]
        return last_sequence.reshape(1, sequence_length, 1)

    def make_prediction(self, ticker, days_ahead=5):

        try:

            model, scaler = self.model_loader.load_model(ticker)
            if model is None or scaler is None:
                return None


            config = self.model_loader.load_config()
            sequence_length = config.get('sequence_length', 60)


            raw_data = self.data_fetcher.get_raw_price_data(ticker, period="6mo")
            if raw_data is None or len(raw_data) < sequence_length:
                st.error(f"Insufficient data for {ticker}")
                return None


            close_prices = raw_data['Close'].values.reshape(-1, 1)


            scaled_data = scaler.transform(close_prices)


            last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)


            predictions = []
            current_sequence = last_sequence.copy()

            for _ in range(days_ahead):

                pred = model.predict(current_sequence, verbose=0)
                predictions.append(pred[0, 0])


                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, 0] = pred[0, 0]


            predictions = np.array(predictions).reshape(-1, 1)
            predictions = scaler.inverse_transform(predictions)


            last_date = raw_data.index[-1]
            prediction_dates = [last_date + timedelta(days=i + 1) for i in range(days_ahead)]


            results = pd.DataFrame({
                'Date': prediction_dates,
                'Predicted_Price': predictions.flatten(),
                'Ticker': ticker
            })


            current_price = raw_data['Close'].iloc[-1]
            predicted_price = predictions[-1][0]
            price_change = predicted_price - current_price
            price_change_pct = (price_change / current_price) * 100


            display_data = self.data_fetcher.get_stock_data(ticker, period="6mo")

            prediction_info = {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'predictions_df': results,
                'historical_data': display_data,
                'raw_data': raw_data,
                'model_info': self.model_loader.get_model_info(ticker)
            }

            return prediction_info

        except Exception as e:
            st.error(f"Prediction failed for {ticker}: {str(e)}")
            return None

    def bulk_predictions(self, tickers, days_ahead=5):
        """Make predictions for multiple tickers"""
        all_predictions = {}

        progress_bar = st.progress(0)
        total_tickers = len(tickers)

        for i, ticker in enumerate(tickers):
            prediction = self.make_prediction(ticker, days_ahead)
            if prediction:
                all_predictions[ticker] = prediction


            progress_bar.progress((i + 1) / total_tickers)

        progress_bar.empty()
        return all_predictions

    def get_prediction_confidence(self, ticker):

        model_info = self.model_loader.get_model_info(ticker)
        if not model_info:
            return "Unknown"


        test_r2 = model_info.get('test_r2', 0)
        test_rmse = model_info.get('test_rmse', float('inf'))


        if test_r2 > 0.8:
            return "High"
        elif test_r2 > 0.6:
            return "Medium"
        elif test_r2 > 0.3:
            return "Low"
        else:
            return "Very Low"

    def get_trading_signals(self, ticker):
        """Generate trading signals based on predictions and technical indicators"""
        try:
            prediction_info = self.make_prediction(ticker, days_ahead=5)
            if not prediction_info:
                return None


            data = prediction_info['historical_data']
            current_price = prediction_info['current_price']
            predicted_price = prediction_info['predicted_price']
            change_pct = prediction_info['price_change_pct']

            signals = []


            if change_pct > 5:
                signals.append("🟢 Strong Buy - BiLSTM Model predicts 5%+ increase")
            elif change_pct > 2:
                signals.append("🟢 Buy - BiLSTM Model predicts 2-5% increase")
            elif change_pct < -5:
                signals.append("🔴 Strong Sell - BiLSTM Model predicts 5%+ decrease")
            elif change_pct < -2:
                signals.append("🔴 Sell - BiLSTM Model predicts 2-5% decrease")
            else:
                signals.append("🟡 Hold - BiLSTM Model predicts minor price movement")


            confidence = self.get_prediction_confidence(ticker)
            signals.append(f"🎯 Model Confidence: {confidence}")


            if not data.empty and len(data) > 50:
                try:
                    latest = data.iloc[-1]

                    # RSI signal
                    rsi = latest.get('RSI')
                    if pd.notna(rsi):
                        if rsi < 30:
                            signals.append("📈 RSI Oversold - Potential bounce")
                        elif rsi > 70:
                            signals.append("📉 RSI Overbought - Potential pullback")

                    # Moving average signal
                    ma_20 = latest.get('MA_20')
                    ma_50 = latest.get('MA_50')

                    if pd.notna(ma_20) and pd.notna(ma_50):
                        if current_price > ma_20 > ma_50:
                            signals.append("📈 Above key moving averages")
                        elif current_price < ma_20 < ma_50:
                            signals.append("📉 Below key moving averages")

                except Exception:

                    pass

            return {
                'ticker': ticker,
                'signals': signals,
                'confidence': confidence,
                'prediction_info': prediction_info
            }

        except Exception as e:
            st.error(f"Failed to generate signals for {ticker}: {str(e)}")
            return None

    def validate_model_performance(self, ticker, test_days=30):

        try:

            raw_data = self.data_fetcher.get_raw_price_data(ticker, period="3mo")
            if raw_data is None or len(raw_data) < test_days + 60:
                return None

            model, scaler = self.model_loader.load_model(ticker)
            if model is None or scaler is None:
                return None


            train_data = raw_data[:-test_days]
            test_data = raw_data[-test_days:]


            close_prices = train_data['Close'].values.reshape(-1, 1)
            scaled_data = scaler.transform(close_prices)

            predictions = []
            for i in range(test_days):
                if len(scaled_data) >= 60:
                    last_sequence = scaled_data[-60:].reshape(1, 60, 1)
                    pred = model.predict(last_sequence, verbose=0)
                    pred_unscaled = scaler.inverse_transform(pred.reshape(-1, 1))[0, 0]
                    predictions.append(pred_unscaled)


                    if i < test_days - 1:
                        actual_next = test_data.iloc[i]['Close']
                        actual_scaled = scaler.transform([[actual_next]])[0, 0]
                        scaled_data = np.append(scaled_data, [[actual_scaled]], axis=0)


            actual_prices = test_data['Close'].values[:len(predictions)]
            predictions = np.array(predictions)

            mse = np.mean((actual_prices - predictions) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(actual_prices - predictions))

            return {
                'rmse': rmse,
                'mae': mae,
                'actual': actual_prices,
                'predicted': predictions,
                'dates': test_data.index[:len(predictions)]
            }

        except Exception as e:
            st.error(f"Validation failed for {ticker}: {str(e)}")
            return None