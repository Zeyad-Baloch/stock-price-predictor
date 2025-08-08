import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta


class PortfolioManager:
    def __init__(self, data_fetcher):
        self.data_fetcher = data_fetcher

    def initialize_portfolio(self):

        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = pd.DataFrame(columns=[
                'Ticker', 'Shares', 'Buy_Price', 'Buy_Date', 'Current_Price',
                'Current_Value', 'Total_Investment', 'P&L', 'P&L_Percent'
            ])

    def add_stock(self, ticker, shares, buy_price, buy_date):

        self.initialize_portfolio()


        current_data = self.data_fetcher.get_stock_data(ticker, period="5d")
        if current_data is None:
            st.error(f"Could not fetch current price for {ticker}")
            return False

        current_price = current_data['Close'].iloc[-1]
        total_investment = shares * buy_price
        current_value = shares * current_price
        pnl = current_value - total_investment
        pnl_percent = (pnl / total_investment) * 100

        new_entry = pd.DataFrame({
            'Ticker': [ticker],
            'Shares': [shares],
            'Buy_Price': [buy_price],
            'Buy_Date': [buy_date],
            'Current_Price': [current_price],
            'Current_Value': [current_value],
            'Total_Investment': [total_investment],
            'P&L': [pnl],
            'P&L_Percent': [pnl_percent]
        })

        st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_entry],
                                               ignore_index=True)
        return True

    def remove_stock(self, index):

        if 'portfolio' in st.session_state and not st.session_state.portfolio.empty:
            st.session_state.portfolio = st.session_state.portfolio.drop(index).reset_index(drop=True)

    def update_portfolio(self):

        if 'portfolio' not in st.session_state or st.session_state.portfolio.empty:
            return

        portfolio = st.session_state.portfolio.copy()

        for i, row in portfolio.iterrows():
            ticker = row['Ticker']
            shares = row['Shares']
            buy_price = row['Buy_Price']
            total_investment = shares * buy_price

            # Get current price
            current_data = self.data_fetcher.get_stock_data(ticker, period="5d")
            if current_data is not None:
                current_price = current_data['Close'].iloc[-1]
                current_value = shares * current_price
                pnl = current_value - total_investment
                pnl_percent = (pnl / total_investment) * 100


                portfolio.at[i, 'Current_Price'] = current_price
                portfolio.at[i, 'Current_Value'] = current_value
                portfolio.at[i, 'P&L'] = pnl
                portfolio.at[i, 'P&L_Percent'] = pnl_percent

        st.session_state.portfolio = portfolio

    def get_portfolio_summary(self):

        if 'portfolio' not in st.session_state or st.session_state.portfolio.empty:
            return None

        portfolio = st.session_state.portfolio

        total_investment = portfolio['Total_Investment'].sum()
        total_current_value = portfolio['Current_Value'].sum()
        total_pnl = portfolio['P&L'].sum()
        total_pnl_percent = (total_pnl / total_investment) * 100 if total_investment > 0 else 0


        best_performer = portfolio.loc[portfolio['P&L_Percent'].idxmax()] if not portfolio.empty else None
        worst_performer = portfolio.loc[portfolio['P&L_Percent'].idxmin()] if not portfolio.empty else None

        return {
            'total_investment': total_investment,
            'total_current_value': total_current_value,
            'total_pnl': total_pnl,
            'total_pnl_percent': total_pnl_percent,
            'num_positions': len(portfolio),
            'best_performer': best_performer,
            'worst_performer': worst_performer,
            'portfolio_data': portfolio
        }

    def get_sector_allocation(self):
        """Get portfolio allocation by sector"""
        if 'portfolio' not in st.session_state or st.session_state.portfolio.empty:
            return None

        portfolio = st.session_state.portfolio
        sector_data = {}

        for _, row in portfolio.iterrows():
            ticker = row['Ticker']
            value = row['Current_Value']

            # stock info to determine sector
            stock_info = self.data_fetcher.get_stock_info(ticker)
            sector = stock_info.get('sector', 'Unknown')

            if sector in sector_data:
                sector_data[sector] += value
            else:
                sector_data[sector] = value

        return sector_data

    def calculate_portfolio_risk(self):

        if 'portfolio' not in st.session_state or st.session_state.portfolio.empty:
            return None

        portfolio = st.session_state.portfolio
        tickers = portfolio['Ticker'].tolist()


        returns_data = {}
        for ticker in tickers:
            data = self.data_fetcher.get_stock_data(ticker, period="1y")
            if data is not None:
                returns = data['Close'].pct_change().dropna()
                returns_data[ticker] = returns

        if not returns_data:
            return None


        returns_df = pd.DataFrame(returns_data)


        weights = []
        total_value = 0
        available_tickers = returns_df.columns.tolist()


        for _, row in portfolio.iterrows():
            if row['Ticker'] in available_tickers:
                total_value += row['Current_Value']


        for ticker in available_tickers:
            stock_row = portfolio[portfolio['Ticker'] == ticker]
            if not stock_row.empty:
                weight = stock_row['Current_Value'].iloc[0] / total_value
                weights.append(weight)

        weights = np.array(weights)


        if len(weights) != len(returns_df.columns):
            return None


        portfolio_returns = (returns_df * weights).sum(axis=1)

        volatility = portfolio_returns.std() * np.sqrt(252)
        avg_return = portfolio_returns.mean() * 252
        sharpe_ratio = avg_return / volatility if volatility > 0 else 0


        var_95 = np.percentile(portfolio_returns, 5)

        return {
            'volatility': volatility,
            'avg_return': avg_return,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95,
            'returns_data': returns_df
        }

    def export_portfolio_data(self):

        if 'portfolio' not in st.session_state or st.session_state.portfolio.empty:
            return None

        portfolio = st.session_state.portfolio.copy()


        portfolio['Export_Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        return portfolio.to_csv(index=False)

    def get_dividend_tracker(self):

        if 'portfolio' not in st.session_state or st.session_state.portfolio.empty:
            return None

        portfolio = st.session_state.portfolio
        dividend_data = []

        for _, row in portfolio.iterrows():
            ticker = row['Ticker']
            shares = row['Shares']

            stock_info = self.data_fetcher.get_stock_info(ticker)
            dividend_yield = stock_info.get('dividend_yield', 0)

            if dividend_yield and dividend_yield > 0:
                annual_dividend = row['Current_Value'] * dividend_yield
                quarterly_dividend = annual_dividend / 4

                dividend_data.append({
                    'Ticker': ticker,
                    'Shares': shares,
                    'Dividend_Yield': f"{dividend_yield * 100:.2f}%",
                    'Annual_Dividend': annual_dividend,
                    'Quarterly_Dividend': quarterly_dividend
                })

        return pd.DataFrame(dividend_data) if dividend_data else None