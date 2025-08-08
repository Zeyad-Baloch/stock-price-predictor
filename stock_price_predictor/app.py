import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os


from utils.model_loader import ModelLoader
from utils.data_fetcher import DataFetcher
from utils.predictor import StockPredictor
from utils.portfolio import PortfolioManager


st.set_page_config(
    page_title="AI Stock Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e1e5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .profit {
        color: #00C851;
    }
    .loss {
        color: #ff4444;
    }
</style>
""", unsafe_allow_html=True)



@st.cache_resource
def init_components():
    model_loader = ModelLoader()
    data_fetcher = DataFetcher()
    predictor = StockPredictor(model_loader, data_fetcher)
    portfolio_manager = PortfolioManager(data_fetcher)
    return model_loader, data_fetcher, predictor, portfolio_manager



def main():

    model_loader, data_fetcher, predictor, portfolio_manager = init_components()


    st.sidebar.title(" 📈Stock Analyzer")
    st.sidebar.markdown("---")


    pages = {
        "🏠 Dashboard": "dashboard",
        "📊 Stock Predictions": "predictions",
        "💼 Portfolio Tracker": "portfolio",
        "📈 Technical Analysis": "analysis",
        "📋 Reports & Export": "reports"
    }

    selected_page = st.sidebar.selectbox("Navigate to:", list(pages.keys()))
    page_key = pages[selected_page]

    # Page routing
    if page_key == "dashboard":
        show_dashboard(model_loader, data_fetcher, predictor, portfolio_manager)
    elif page_key == "predictions":
        show_predictions_page(model_loader, data_fetcher, predictor)
    elif page_key == "portfolio":
        show_portfolio_page(portfolio_manager, data_fetcher, predictor)
    elif page_key == "analysis":
        show_analysis_page(data_fetcher)
    elif page_key == "reports":
        show_reports_page(portfolio_manager, predictor)


def show_dashboard(model_loader, data_fetcher, predictor, portfolio_manager):

    st.title("Stock Market Dashboard")
    st.markdown("Welcome to your intelligent stock market companion!")

    portfolio_summary = portfolio_manager.get_portfolio_summary()


    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(" Portfolio Overview")
        if portfolio_summary:

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.metric("Total Value", f"${portfolio_summary['total_current_value']:.2f}")

            with metric_col2:
                pnl_delta = f"{portfolio_summary['total_pnl_percent']:+.2f}%"
                st.metric("Total P&L",
                          f"${portfolio_summary['total_pnl']:+.2f}",
                          pnl_delta)

            with metric_col3:
                st.metric("Positions", portfolio_summary['num_positions'])


            if not portfolio_summary['portfolio_data'].empty:
                portfolio_data = portfolio_summary['portfolio_data']


                fig_performance = go.Figure()

                colors = ['green' if x >= 0 else 'red' for x in portfolio_data['P&L_Percent']]

                fig_performance.add_trace(go.Bar(
                    x=portfolio_data['Ticker'],
                    y=portfolio_data['P&L_Percent'],
                    marker_color=colors,
                    text=[f"{x:+.1f}%" for x in portfolio_data['P&L_Percent']],
                    textposition='outside',
                    name="Performance %"
                ))

                fig_performance.update_layout(
                    title="Portfolio Performance by Stock",
                    xaxis_title="Stocks",
                    yaxis_title="Performance (%)",
                    height=300,
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )

                st.plotly_chart(fig_performance, use_container_width=True)
        else:
            st.info(" No portfolio data yet. Add stocks in Portfolio Tracker to see your performance!")


            sample_data = {
                'Ticker': ['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
                'Performance': [5.2, -2.1, 3.8, -0.5]
            }

            fig_sample = go.Figure()
            colors = ['green' if x >= 0 else 'red' for x in sample_data['Performance']]

            fig_sample.add_trace(go.Bar(
                x=sample_data['Ticker'],
                y=sample_data['Performance'],
                marker_color=colors,
                text=[f"{x:+.1f}%" for x in sample_data['Performance']],
                textposition='outside',
                name="Sample Performance",
                opacity=0.6
            ))

            fig_sample.update_layout(
                title="Sample Portfolio Performance (Demo)",
                xaxis_title="Stocks",
                yaxis_title="Performance (%)",
                height=300,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig_sample, use_container_width=True)

    with col2:
        st.subheader(" Market Overview")


        try:
            market_data = data_fetcher.get_market_indices()

            for name, data in market_data.items():
                change_color = "🟢" if data['change'] >= 0 else "🔴"
                st.metric(
                    name,
                    f"{data['value']:.2f}",
                    f"{change_color} {data['change']:+.2f} ({data['change_pct']:+.2f}%)"
                )
        except:

            demo_market = {
                'S&P 500': {'value': 4567.89, 'change': 12.34, 'change_pct': 0.27},
                'NASDAQ': {'value': 14234.56, 'change': -23.45, 'change_pct': -0.16},
                'DOW JONES': {'value': 34567.12, 'change': 45.67, 'change_pct': 0.13}
            }

            for name, data in demo_market.items():
                change_color = "🟢" if data['change'] >= 0 else "🔴"
                st.metric(
                    name,
                    f"{data['value']:.2f}",
                    f"{change_color} {data['change']:+.2f} ({data['change_pct']:+.2f}%)"
                )


    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(" Portfolio Allocation")

        if portfolio_summary and not portfolio_summary['portfolio_data'].empty:
            portfolio_data = portfolio_summary['portfolio_data']

            fig_pie = px.pie(
                values=portfolio_data['Current_Value'],
                names=portfolio_data['Ticker'],
                title="Current Portfolio Allocation"
            )

            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(height=400)

            st.plotly_chart(fig_pie, use_container_width=True)
        else:

            demo_allocation = {
                'Technology': 40,
                'Healthcare': 25,
                'Finance': 20,
                'Energy': 10,
                'Others': 5
            }

            fig_demo_pie = px.pie(
                values=list(demo_allocation.values()),
                names=list(demo_allocation.keys()),
                title="Sample Sector Allocation "
            )

            fig_demo_pie.update_traces(textposition='inside', textinfo='percent+label', opacity=0.7)
            fig_demo_pie.update_layout(height=400)

            st.plotly_chart(fig_demo_pie, use_container_width=True)

    with col2:
        st.subheader(" Top Performers")

        if portfolio_summary and not portfolio_summary['portfolio_data'].empty:
            portfolio_data = portfolio_summary['portfolio_data']


            top_performers = portfolio_data.nlargest(3, 'P&L_Percent')

            for _, stock in top_performers.iterrows():
                col_stock, col_perf = st.columns([2, 1])

                with col_stock:
                    st.markdown(f"**{stock['Ticker']}**")
                    st.markdown(f"${stock['Current_Price']:.2f}")

                with col_perf:
                    perf_color = "🟢" if stock['P&L_Percent'] >= 0 else "🔴"
                    st.markdown(f"{perf_color} **{stock['P&L_Percent']:+.2f}%**")
                    st.markdown(f"${stock['P&L']:+.2f}")
        else:

            demo_performers = [
                {'ticker': 'AAPL', 'price': 175.43, 'change': 5.2, 'pnl': 245.67},
                {'ticker': 'MSFT', 'price': 332.89, 'change': 3.8, 'pnl': 189.23},
                {'ticker': 'GOOGL', 'price': 2543.21, 'change': -2.1, 'pnl': -67.45}
            ]

            for stock in demo_performers:
                col_stock, col_perf = st.columns([2, 1])

                with col_stock:
                    st.markdown(f"**{stock['ticker']}** (Demo)")
                    st.markdown(f"${stock['price']:.2f}")

                with col_perf:
                    perf_color = "🟢" if stock['change'] >= 0 else "🔴"
                    st.markdown(f"{perf_color} **{stock['change']:+.2f}%**")
                    st.markdown(f"${stock['pnl']:+.2f}")

        st.markdown("###  Risk Metrics")


        try:
            risk_data = portfolio_manager.calculate_portfolio_risk()
            if risk_data:
                st.metric("Volatility", f"{risk_data['volatility'] * 100:.2f}%")
                st.metric("Sharpe Ratio", f"{risk_data['sharpe_ratio']:.2f}")
                st.metric("VaR (95%)", f"{risk_data['var_95'] * 100:.2f}%")
        except:

            st.metric("Volatility", "12.34%")
            st.metric("Sharpe Ratio", "1.23")
            st.metric("VaR (95%)", "-2.45%")


    st.markdown("---")

    st.markdown("---")
    st.subheader(" Quick Stats")

    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

    with stat_col1:
        st.metric("Active Positions", portfolio_summary['num_positions'] if portfolio_summary else 0)

    with stat_col2:
        st.metric("Market Status", "🟢 Open" if datetime.now().weekday() < 5 else "🔴 Closed")

    with stat_col3:
        st.metric("Data Last Updated", datetime.now().strftime("%H:%M"))

    with stat_col4:
        portfolio_value = portfolio_summary['total_current_value'] if portfolio_summary else 0
        st.metric("Portfolio Tier",
                  "🥇 Gold" if portfolio_value > 100000 else
                  "🥈 Silver" if portfolio_value > 50000 else
                  "🥉 Bronze" if portfolio_value > 10000 else
                  "🎯 Starter")


def show_predictions_page(model_loader, data_fetcher, predictor):
    """Stock predictions page"""
    st.title(" Stock Predictions")

    available_models = model_loader.list_available_models()

    if not available_models:
        st.error(" No trained models found!")
        return


    col1, col2 = st.columns([2, 1])

    with col1:
        selected_ticker = st.selectbox("Select Stock:", available_models)

    with col2:
        days_ahead = st.slider("Prediction Days", 1, 30, 5)

    if st.button("Generate Predictions", use_container_width=True):
        with st.spinner(f"Analyzing {selected_ticker}..."):
            prediction = predictor.make_prediction(selected_ticker, days_ahead)

            if prediction:

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Current Price", f"${prediction['current_price']:.2f}")

                with col2:
                    st.metric(f"{days_ahead}-Day Prediction", f"${prediction['predicted_price']:.2f}")

                with col3:
                    change_color = "profit" if prediction['price_change'] >= 0 else "loss"
                    st.markdown(f"<p class='{change_color}'>Expected Change: ${prediction['price_change']:+.2f}</p>",
                                unsafe_allow_html=True)

                with col4:
                    change_color = "profit" if prediction['price_change_pct'] >= 0 else "loss"
                    st.markdown(f"<p class='{change_color}'>Change %: {prediction['price_change_pct']:+.2f}%</p>",
                                unsafe_allow_html=True)

                # Prediction chart
                st.subheader("📈 Price Prediction Chart")

                fig = go.Figure()


                historical = prediction['historical_data'].tail(60)
                fig.add_trace(go.Scatter(
                    x=historical.index,
                    y=historical['Close'],
                    mode='lines',
                    name='Historical Price',
                    line=dict(color='blue')
                ))


                pred_df = prediction['predictions_df']
                fig.add_trace(go.Scatter(
                    x=pred_df['Date'],
                    y=pred_df['Predicted_Price'],
                    mode='lines+markers',
                    name='Predicted Price',
                    line=dict(color='red', dash='dash')
                ))

                fig.update_layout(
                    title=f"{selected_ticker} - Price Prediction",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)


                signals = predictor.get_trading_signals(selected_ticker)
                if signals:
                    st.subheader(" Trading Signals")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Signals:**")
                        for signal in signals['signals']:
                            st.markdown(f"• {signal}")

                    with col2:
                        confidence = signals['confidence']
                        confidence_color = {
                            'High': '🟢',
                            'Medium': '🟡',
                            'Low': '🔴'
                        }.get(confidence, '⚪')
                        st.markdown(f"**Confidence:** {confidence_color} {confidence}")


    st.markdown("---")
    st.subheader("Multiple Analysis")

    if st.checkbox("Enable Multiple predictions"):
        num_stocks = st.slider("Number of stocks to analyze", 3, min(10, len(available_models)), 5)
        selected_stocks = st.multiselect("Select stocks:", available_models,
                                         default=available_models[:num_stocks])

        if st.button(" Run ") and selected_stocks:
            bulk_predictions = predictor.bulk_predictions(selected_stocks, days_ahead=5)

            if bulk_predictions:

                summary_data = []
                for ticker, pred in bulk_predictions.items():
                    summary_data.append({
                        'Ticker': ticker,
                        'Current': f"${pred['current_price']:.2f}",
                        'Predicted': f"${pred['predicted_price']:.2f}",
                        'Change': f"{pred['price_change_pct']:+.2f}%",
                        'Signal': '🟢 Buy' if pred['price_change_pct'] > 2 else '🔴 Sell' if pred[
                                                                                               'price_change_pct'] < -2 else '🟡 Hold'
                    })

                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)


def show_portfolio_page(portfolio_manager, data_fetcher, predictor):
    """Portfolio tracking page"""
    st.title(" Portfolio Tracker")


    portfolio_summary = portfolio_manager.get_portfolio_summary()

    if portfolio_summary:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Investment", f"${portfolio_summary['total_investment']:.2f}")

        with col2:
            st.metric("Current Value", f"${portfolio_summary['total_current_value']:.2f}")

        with col3:
            pnl_color = "profit" if portfolio_summary['total_pnl'] >= 0 else "loss"
            st.markdown(
                f"<div class='stMetric'><p class='{pnl_color}'>Total P&L: ${portfolio_summary['total_pnl']:+.2f}</p></div>",
                unsafe_allow_html=True)

        with col4:
            pnl_pct_color = "profit" if portfolio_summary['total_pnl_percent'] >= 0 else "loss"
            st.markdown(
                f"<div class='stMetric'><p class='{pnl_pct_color}'>Return: {portfolio_summary['total_pnl_percent']:+.2f}%</p></div>",
                unsafe_allow_html=True)


    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("➕ Add Stock")
        with st.form("add_stock_form"):
            ticker = st.text_input("Ticker Symbol").upper()
            shares = st.number_input("Number of Shares", min_value=1, value=100)
            buy_price = st.number_input("Buy Price ($)", min_value=0.01, value=100.0, step=0.01)
            buy_date = st.date_input("Buy Date", value=datetime.now().date())

            if st.form_submit_button("Add to Portfolio"):
                if ticker:
                    if portfolio_manager.add_stock(ticker, shares, buy_price, buy_date):
                        st.success(f"✅ Added {shares} shares of {ticker}")
                        st.rerun()

    with col2:
        st.subheader(" Update")
        if st.button("Refresh Prices", use_container_width=True):
            portfolio_manager.update_portfolio()
            st.success("Portfolio updated!")
            st.rerun()

        if portfolio_summary and not portfolio_summary['portfolio_data'].empty:
            st.subheader(" ")
            portfolio_data = portfolio_summary['portfolio_data']
            remove_options = [f"{row['Ticker']} ({row['Shares']} shares)"
                              for _, row in portfolio_data.iterrows()]

            if remove_options:
                selected_remove = st.selectbox("Select to remove:", remove_options)
                if st.button("Remove Stock"):
                    remove_index = remove_options.index(selected_remove)
                    portfolio_manager.remove_stock(remove_index)
                    st.success("Stock removed!")
                    st.rerun()


    if portfolio_summary and not portfolio_summary['portfolio_data'].empty:
        st.markdown("---")
        st.subheader("📊 Your Portfolio")


        display_df = portfolio_summary['portfolio_data'].copy()
        display_df['Buy_Price'] = display_df['Buy_Price'].apply(lambda x: f"${x:.2f}")
        display_df['Current_Price'] = display_df['Current_Price'].apply(lambda x: f"${x:.2f}")
        display_df['Current_Value'] = display_df['Current_Value'].apply(lambda x: f"${x:.2f}")
        display_df['Total_Investment'] = display_df['Total_Investment'].apply(lambda x: f"${x:.2f}")
        display_df['P&L'] = display_df['P&L'].apply(lambda x: f"${x:+.2f}")
        display_df['P&L_Percent'] = display_df['P&L_Percent'].apply(lambda x: f"{x:+.2f}%")

        st.dataframe(display_df, use_container_width=True)

        # Portfolio visualization
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Portfolio Allocation")
            portfolio_data = portfolio_summary['portfolio_data']

            fig = px.pie(
                values=portfolio_data['Current_Value'],
                names=portfolio_data['Ticker'],
                title="Portfolio Allocation by Value"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("💹 Performance Chart")


            fig = go.Figure()

            tickers = portfolio_data['Ticker'].tolist()
            pnl_values = portfolio_data['P&L_Percent'].tolist()
            colors = ['green' if x >= 0 else 'red' for x in pnl_values]

            fig.add_trace(go.Bar(
                x=tickers,
                y=pnl_values,
                marker_color=colors,
                name="P&L %"
            ))

            fig.update_layout(
                title="Stock Performance (%)",
                xaxis_title="Stocks",
                yaxis_title="P&L Percentage"
            )

            st.plotly_chart(fig, use_container_width=True)


        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            if portfolio_summary['best_performer'] is not None:
                best = portfolio_summary['best_performer']
                st.success(f"Best Performer: {best['Ticker']} (+{best['P&L_Percent']:.2f}%)")

        with col2:
            if portfolio_summary['worst_performer'] is not None:
                worst = portfolio_summary['worst_performer']
                st.error(f"Worst Performer: {worst['Ticker']} ({worst['P&L_Percent']:.2f}%)")


        risk_data = portfolio_manager.calculate_portfolio_risk()
        if risk_data:
            st.markdown("---")
            st.subheader("Risk Metrics")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Volatility", f"{risk_data['volatility'] * 100:.2f}%")
            with col2:
                st.metric("Expected Return", f"{risk_data['avg_return'] * 100:.2f}%")
            with col3:
                st.metric("Sharpe Ratio", f"{risk_data['sharpe_ratio']:.2f}")
            with col4:
                st.metric("VaR (95%)", f"{risk_data['var_95'] * 100:.2f}%")


def show_analysis_page(data_fetcher):
    """Technical analysis page"""
    st.title(" Technical Analysis")

    # Stock selection
    ticker = st.text_input("Enter Stock Symbol:", value="AAPL").upper()
    period = st.selectbox("Time Period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])

    if ticker:
        data = data_fetcher.get_stock_data(ticker, period=period)

        if data is not None and not data.empty:

            stock_info = data_fetcher.get_stock_info(ticker)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Company", stock_info.get('name', ticker))
            with col2:
                st.metric("Current Price", f"${stock_info.get('current_price', 0):.2f}")
            with col3:
                st.metric("Market Cap", f"${stock_info.get('market_cap', 0):,.0f}")
            with col4:
                st.metric("P/E Ratio", f"{stock_info.get('pe_ratio', 0):.2f}")


            st.subheader("📊 Price Chart with Technical Indicators")

            fig = go.Figure()


            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Price"
            ))

            # Moving averages
            if 'MA_20' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['MA_20'],
                    mode='lines',
                    name='MA 20',
                    line=dict(color='orange')
                ))

            if 'MA_50' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['MA_50'],
                    mode='lines',
                    name='MA 50',
                    line=dict(color='red')
                ))

            # Bollinger Bands
            if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['BB_Upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='gray', dash='dash')
                ))

                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['BB_Lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='gray', dash='dash'),
                    fill='tonexty'
                ))

            fig.update_layout(
                title=f"{ticker} - Technical Analysis",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=600,
                xaxis_rangeslider_visible=False
            )

            st.plotly_chart(fig, use_container_width=True)


            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 RSI (Relative Strength Index)")
                if 'RSI' in data.columns:
                    latest_rsi = data['RSI'].iloc[-1]

                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(
                        x=data.index,
                        y=data['RSI'],
                        mode='lines',
                        name='RSI'
                    ))

                    # RSI levels
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")

                    fig_rsi.update_layout(
                        title=f"RSI: {latest_rsi:.2f}",
                        xaxis_title="Date",
                        yaxis_title="RSI",
                        height=300
                    )

                    st.plotly_chart(fig_rsi, use_container_width=True)

                    # RSI interpretation
                    if latest_rsi > 70:
                        st.warning("⚠️ Overbought - Potential sell signal")
                    elif latest_rsi < 30:
                        st.success("Oversold - Potential buy signal")
                    else:
                        st.info("Neutral territory")

            with col2:
                st.subheader("📈 MACD")
                if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                    fig_macd = go.Figure()

                    fig_macd.add_trace(go.Scatter(
                        x=data.index,
                        y=data['MACD'],
                        mode='lines',
                        name='MACD'
                    ))

                    fig_macd.add_trace(go.Scatter(
                        x=data.index,
                        y=data['MACD_Signal'],
                        mode='lines',
                        name='Signal'
                    ))

                    # MACD histogram
                    histogram = data['MACD'] - data['MACD_Signal']
                    fig_macd.add_trace(go.Bar(
                        x=data.index,
                        y=histogram,
                        name='Histogram',
                        opacity=0.6
                    ))

                    fig_macd.update_layout(
                        title="MACD",
                        xaxis_title="Date",
                        yaxis_title="MACD",
                        height=300
                    )

                    st.plotly_chart(fig_macd, use_container_width=True)

            # Volume analysis
            st.subheader("📊 Volume Analysis")

            fig_volume = go.Figure()

            fig_volume.add_trace(go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume'
            ))

            # Volume moving average
            volume_ma = data['Volume'].rolling(window=20).mean()
            fig_volume.add_trace(go.Scatter(
                x=data.index,
                y=volume_ma,
                mode='lines',
                name='Volume MA 20',
                line=dict(color='red')
            ))

            fig_volume.update_layout(
                title="Volume Analysis",
                xaxis_title="Date",
                yaxis_title="Volume",
                height=400
            )

            st.plotly_chart(fig_volume, use_container_width=True)

        else:
            st.error(f"Could not fetch data for {ticker}")


def show_reports_page(portfolio_manager, predictor):
    """Reports and export page"""
    st.title("Reports & Export")

    st.subheader("Portfolio Reports")

    portfolio_summary = portfolio_manager.get_portfolio_summary()

    if portfolio_summary:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📊 Portfolio Summary Report**")


            report_data = {
                "Total Investment": f"${portfolio_summary['total_investment']:.2f}",
                "Current Value": f"${portfolio_summary['total_current_value']:.2f}",
                "Total P&L": f"${portfolio_summary['total_pnl']:+.2f}",
                "Return %": f"{portfolio_summary['total_pnl_percent']:+.2f}%",
                "Number of Positions": portfolio_summary['num_positions'],
                "Report Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            for key, value in report_data.items():
                st.markdown(f"**{key}:** {value}")

        with col2:
            st.markdown("**💰 Dividend Report**")

            dividend_data = portfolio_manager.get_dividend_tracker()
            if dividend_data is not None and not dividend_data.empty:
                st.dataframe(dividend_data, use_container_width=True)

                total_annual_dividend = dividend_data['Annual_Dividend'].sum()
                st.metric("Total Annual Dividends", f"${total_annual_dividend:.2f}")
            else:
                st.info("No dividend-paying stocks in portfolio")


        st.markdown("---")
        st.subheader("📤 Export Options")

        col1, col2, = st.columns(2)

        with col1:
            if st.button(" Export Portfolio CSV"):
                csv_data = portfolio_manager.export_portfolio_data()
                if csv_data:
                    st.download_button(
                        label="Download Portfolio CSV",
                        data=csv_data,
                        file_name=f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

        with col2:
            if st.button(" Export Performance Report"):

                performance_report = f"""
PORTFOLIO PERFORMANCE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY:
- Total Investment: ${portfolio_summary['total_investment']:.2f}
- Current Value: ${portfolio_summary['total_current_value']:.2f}
- Total P&L: ${portfolio_summary['total_pnl']:+.2f}
- Return Percentage: {portfolio_summary['total_pnl_percent']:+.2f}%
- Number of Positions: {portfolio_summary['num_positions']}

INDIVIDUAL POSITIONS:
"""
                for _, row in portfolio_summary['portfolio_data'].iterrows():
                    performance_report += f"""
{row['Ticker']}:
  Shares: {row['Shares']}
  Buy Price: ${row['Buy_Price']:.2f}
  Current Price: ${row['Current_Price']:.2f}
  P&L: ${row['P&L']:+.2f} ({row['P&L_Percent']:+.2f}%)
"""

                st.download_button(
                    label="Download Performance Report",
                    data=performance_report,
                    file_name=f"performance_report_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )



    else:
        st.info("📝 No portfolio data available for reports")


    st.markdown("---")
    st.subheader("ℹ️ System Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Dashboard Statistics**")
        st.markdown("•  Data Source: Yahoo Finance")
        st.markdown("•  Update Frequency: Real-time")
        st.markdown("•  Portfolio Tracking: Active")

    with col2:
        st.markdown("**Technical Details**")
        st.markdown("•  Framework: Streamlit")
        st.markdown("•  ML Backend: TensorFlow/Keras")
        st.markdown("•  Visualization: Plotly")


if __name__ == "__main__":
    main()

