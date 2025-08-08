import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta


class DataFetcher:
    def __init__(self):
        pass

    @st.cache_data(ttl=300)
    def get_stock_data(_self, ticker, period="1y"):

        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)

            if data.empty:
                return None


            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)


            data = _self.add_technical_indicators_for_display(data)
            return data

        except Exception as e:
            st.error(f"Failed to fetch data for {ticker}: {str(e)}")
            return None

    def add_technical_indicators_for_display(self, data):

        try:
            # Moving averages
            data['MA_20'] = data['Close'].rolling(window=20).mean()
            data['MA_50'] = data['Close'].rolling(window=50).mean()
            data['MA_200'] = data['Close'].rolling(window=200).mean()

            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))

            # MACD
            ema_12 = data['Close'].ewm(span=12).mean()
            ema_26 = data['Close'].ewm(span=26).mean()
            data['MACD'] = ema_12 - ema_26
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()

            # Bollinger Bands
            rolling_mean = data['Close'].rolling(window=20).mean()
            rolling_std = data['Close'].rolling(window=20).std()
            data['BB_Upper'] = rolling_mean + (rolling_std * 2)
            data['BB_Lower'] = rolling_mean - (rolling_std * 2)

        except Exception as e:

            st.warning(f"Some technical indicators couldn't be calculated: {str(e)}")

        return data

    def get_raw_price_data(self, ticker, period="1y"):

        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)

            if data.empty:
                return None


            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)

            return data

        except Exception as e:
            st.error(f"Failed to fetch raw data for {ticker}: {str(e)}")
            return None

    def get_stock_info(self, ticker):

        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                'name': info.get('longName', ticker),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                'current_price': info.get('currentPrice', 0),
                'previous_close': info.get('previousClose', 0)
            }
        except Exception as e:
            return {'name': ticker, 'error': str(e)}

    def get_multiple_stocks_data(self, tickers, period="1y"):

        all_data = {}
        for ticker in tickers:
            data = self.get_stock_data(ticker, period)
            if data is not None:
                all_data[ticker] = data
        return all_data

    def get_market_indices(self):

        indices = {
            'S&P 500': '^GSPC',
            'NASDAQ': '^IXIC',
            'Dow Jones': '^DJI',
            'VIX': '^VIX'
        }

        index_data = {}
        for name, symbol in indices.items():
            try:
                data = yf.Ticker(symbol).history(period="5d")
                if not data.empty:
                    current = data['Close'].iloc[-1]
                    previous = data['Close'].iloc[-2] if len(data) > 1 else current
                    change = current - previous
                    change_pct = (change / previous) * 100

                    index_data[name] = {
                        'value': current,
                        'change': change,
                        'change_pct': change_pct
                    }
            except:
                continue

        return index_data