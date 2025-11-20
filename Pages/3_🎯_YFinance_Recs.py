import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import ta
from sklearn.preprocessing import MinMaxScaler
import json
import os
import hashlib

st.set_page_config(page_title="YFinance Recommendations", page_icon="🎯", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .recommendation-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .recommendation-card.sell {
        border-left-color: #dc3545;
    }
    .recommendation-card.hold {
        border-left-color: #ffc107;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_stock_data(symbol, period="1y"):
    """Get comprehensive stock data"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        info = stock.info
        
        if hist.empty:
            return None, None
        
        return hist, info
    except:
        return None, None

def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Histogram'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'], window=20)
    df['BB_Upper'] = bollinger.bollinger_hband()
    df['BB_Lower'] = bollinger.bollinger_lband()
    df['BB_Middle'] = bollinger.bollinger_mavg()
    
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # EMA
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    return df

def generate_recommendation_score(df, info):
    """Generate recommendation score based on technical analysis"""
    latest = df.iloc[-1]
    scores = []
    signals = []
    
    # RSI Analysis
    rsi = latest['RSI']
    if rsi < 30:
        scores.append(2)  # Oversold - Buy signal
        signals.append("RSI Oversold (Buy)")
    elif rsi > 70:
        scores.append(-2)  # Overbought - Sell signal
        signals.append("RSI Overbought (Sell)")
    else:
        scores.append(0)
        signals.append("RSI Neutral")
    
    # Moving Average Analysis
    price = latest['Close']
    sma_20 = latest['SMA_20']
    sma_50 = latest['SMA_50']
    
    if price > sma_20 and sma_20 > sma_50:
        scores.append(1)
        signals.append("Price above MAs (Bullish)")
    elif price < sma_20 and sma_20 < sma_50:
        scores.append(-1)
        signals.append("Price below MAs (Bearish)")
    else:
        scores.append(0)
        signals.append("MA Mixed signals")
    
    # MACD Analysis
    macd = latest['MACD']
    macd_signal = latest['MACD_Signal']
    
    if macd > macd_signal:
        scores.append(1)
        signals.append("MACD Bullish")
    else:
        scores.append(-1)
        signals.append("MACD Bearish")
    
    # Bollinger Bands
    bb_upper = latest['BB_Upper']
    bb_lower = latest['BB_Lower']
    
    if price < bb_lower:
        scores.append(1)
        signals.append("Below BB Lower (Oversold)")
    elif price > bb_upper:
        scores.append(-1)
        signals.append("Above BB Upper (Overbought)")
    else:
        scores.append(0)
        signals.append("Within BB Range")
    
    # Volume Analysis
    volume_ratio = latest['Volume_Ratio']
    if volume_ratio > 1.5:
        scores.append(0.5)
        signals.append("High Volume")
    elif volume_ratio < 0.5:
        scores.append(-0.5)
        signals.append("Low Volume")
    else:
        scores.append(0)
        signals.append("Normal Volume")
    
    # P/E Ratio (if available)
    pe_ratio = info.get('trailingPE')
    if pe_ratio:
        if pe_ratio < 15:
            scores.append(1)
            signals.append("Low P/E (Undervalued)")
        elif pe_ratio > 25:
            scores.append(-0.5)
            signals.append("High P/E (Overvalued)")
        else:
            scores.append(0)
            signals.append("Moderate P/E")
    
    total_score = sum(scores)
    max_possible = len(scores) * 2
    normalized_score = (total_score + max_possible) / (2 * max_possible)  # Normalize to 0-1
    
    return normalized_score, signals

def get_recommendation_text(score):
    """Convert score to recommendation text"""
    if score >= 0.7:
        return "STRONG BUY", "🟢"
    elif score >= 0.6:
        return "BUY", "🟢"
    elif score >= 0.4:
        return "HOLD", "🟡"
    elif score >= 0.3:
        return "SELL", "🔴"
    else:
        return "STRONG SELL", "🔴"

def create_technical_chart(df, symbol):
    """Create comprehensive technical analysis chart"""
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{symbol} Price & Moving Averages', 'Volume', 'RSI', 'MACD'),
        row_width=[0.2, 0.1, 0.1, 0.1]
    )
    
    # Price and Moving Averages
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(color='gray', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(color='gray', dash='dash')), row=1, col=1)
    
    # Volume
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='lightblue'), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='MACD Signal', line=dict(color='red')), row=4, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Histogram'], name='MACD Histogram', marker_color='gray'), row=4, col=1)
    
    fig.update_layout(height=800, showlegend=True, title_text=f"Technical Analysis - {symbol}")
    return fig

def get_popular_stocks():
    """Get list of popular stocks for analysis"""
    return {
        "Technology": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX"],
        "Finance": ["JPM", "BAC", "WFC", "GS", "MS", "V", "MA"],
        "Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK"],
        "Consumer": ["KO", "PEP", "PG", "WMT", "HD", "MCD"],
        "Energy": ["XOM", "CVX", "COP", "EOG"],
        "Industrial": ["BA", "CAT", "GE", "MMM"]
    }

def init_session_state():
    """Initialize session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None

def check_authentication():
    init_session_state()
    if not st.session_state.authenticated:
        st.warning("Please login from the Homepage to access this feature.")
        st.stop()

def main():
    check_authentication()
    st.markdown('<h1 class="main-header">🎯 YFinance Stock Recommendations</h1>', unsafe_allow_html=True)
    
    # Input section
    st.markdown("## 📊 Stock Analysis & Recommendations")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        symbol = st.text_input("Enter Stock Symbol", value="AAPL", help="Enter a valid stock ticker symbol")
    
    with col2:
        period = st.selectbox("Analysis Period", ["3mo", "6mo", "1y", "2y"], index=2)
    
    with col3:
        investment_amount = st.number_input("Investment Amount ($)", min_value=100, value=10000, step=100)
    
    # Popular stocks selection
    st.markdown("### 🔥 Popular Stocks")
    popular_stocks = get_popular_stocks()
    
    selected_category = st.selectbox("Select Category", list(popular_stocks.keys()))
    selected_stocks = st.multiselect(
        "Or select from popular stocks",
        popular_stocks[selected_category],
        help="Select multiple stocks for batch analysis"
    )
    
    # Analysis button
    analyze_button = st.button("🎯 Analyze Stock(s)", type="primary")
    
    if analyze_button:
        stocks_to_analyze = [symbol.upper()] if symbol and not selected_stocks else [s.upper() for s in selected_stocks]
        
        if not stocks_to_analyze:
            st.error("Please enter a stock symbol or select stocks from the list")
            return
        
        # Progress bar for multiple stocks
        if len(stocks_to_analyze) > 1:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        recommendations = []
        
        for i, stock_symbol in enumerate(stocks_to_analyze):
            if len(stocks_to_analyze) > 1:
                progress_bar.progress((i + 1) / len(stocks_to_analyze))
                status_text.text(f"Analyzing {stock_symbol}...")
            
            # Get stock data
            with st.spinner(f"Fetching data for {stock_symbol}..."):
                hist_data, stock_info = get_stock_data(stock_symbol, period)
            
            if hist_data is None:
                st.error(f"Could not fetch data for {stock_symbol}. Please check the symbol.")
                continue
            
            # Calculate technical indicators
            df_with_indicators = calculate_technical_indicators(hist_data.copy())
            
            # Generate recommendation
            score, signals = generate_recommendation_score(df_with_indicators, stock_info)
            recommendation, emoji = get_recommendation_text(score)
            
            current_price = df_with_indicators['Close'].iloc[-1]
            max_shares = int(investment_amount / current_price)
            
            recommendations.append({
                'symbol': stock_symbol,
                'recommendation': recommendation,
                'emoji': emoji,
                'score': score,
                'signals': signals,
                'current_price': current_price,
                'max_shares': max_shares,
                'investment_value': max_shares * current_price,
                'stock_info': stock_info,
                'df': df_with_indicators
            })
        
        if len(stocks_to_analyze) > 1:
            progress_bar.empty()
            status_text.empty()
        
        # Display recommendations
        st.markdown("## 📈 Analysis Results")
        
        # Summary for multiple stocks
        if len(recommendations) > 1:
            buy_count = sum(1 for r in recommendations if 'BUY' in r['recommendation'])
            hold_count = sum(1 for r in recommendations if 'HOLD' in r['recommendation'])
            sell_count = sum(1 for r in recommendations if 'SELL' in r['recommendation'])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Stocks Analyzed", len(recommendations))
            with col2:
                st.metric("Buy Signals", buy_count)
            with col3:
                st.metric("Hold Signals", hold_count)
            with col4:
                st.metric("Sell Signals", sell_count)
        
        # Individual stock analysis
        for rec in recommendations:
            st.markdown(f"### {rec['emoji']} {rec['symbol']} - {rec['recommendation']}")
            
            # Stock info
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"${rec['current_price']:.2f}")
            with col2:
                st.metric("Recommendation Score", f"{rec['score']:.2f}")
            with col3:
                st.metric("Max Shares", f"{rec['max_shares']:,}")
            with col4:
                st.metric("Investment Value", f"${rec['investment_value']:,.2f}")
            
            # Company info
            if rec['stock_info']:
                info = rec['stock_info']
                st.markdown(f"**Company:** {info.get('longName', rec['symbol'])}")
                st.markdown(f"**Sector:** {info.get('sector', 'N/A')} | **Industry:** {info.get('industry', 'N/A')}")
                
                info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                with info_col1:
                    st.metric("Market Cap", f"${info.get('marketCap', 0):,.0f}")
                with info_col2:
                    st.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
                with info_col3:
                    st.metric("52W High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
                with info_col4:
                    st.metric("52W Low", f"${info.get('fiftyTwoWeekLow', 'N/A')}")
            
            # Technical signals
            with st.expander(f"📊 Technical Analysis Details - {rec['symbol']}"):
                st.markdown("**Technical Signals:**")
                for signal in rec['signals']:
                    st.write(f"• {signal}")
                
                # Technical chart
                tech_chart = create_technical_chart(rec['df'], rec['symbol'])
                st.plotly_chart(tech_chart, use_container_width=True)
            
            st.divider()
        
        # Portfolio recommendation
        if len(recommendations) > 1:
            st.markdown("## 💼 Portfolio Recommendation")
            
            # Filter buy recommendations
            buy_recs = [r for r in recommendations if 'BUY' in r['recommendation']]
            
            if buy_recs:
                # Sort by score
                buy_recs.sort(key=lambda x: x['score'], reverse=True)
                
                st.markdown("### 🟢 Recommended Portfolio Allocation")
                
                total_investment = sum(r['investment_value'] for r in buy_recs)
                
                for i, rec in enumerate(buy_recs[:5], 1):  # Top 5 buy recommendations
                    allocation = (rec['investment_value'] / total_investment) * 100
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <h5>#{i}. {rec['symbol']} - {rec['recommendation']}</h5>
                        <p><strong>Allocation:</strong> {allocation:.1f}% (${rec['investment_value']:,.2f})</p>
                        <p><strong>Shares:</strong> {rec['max_shares']:,} @ ${rec['current_price']:.2f}</p>
                        <p><strong>Score:</strong> {rec['score']:.2f}/1.0</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No buy recommendations found in the analyzed stocks.")
    
    # Market overview
    st.markdown("## 📊 Quick Market Overview")
    st.info("💡 **Pro Tip:** Combine technical analysis with fundamental research and market conditions for better investment decisions!")
    
    # Disclaimer
    st.markdown("## ⚠️ Important Disclaimer")
    st.warning("""
    **Investment Risk Warning:** These recommendations are based on technical analysis and historical data only. 
    Stock market investments carry inherent risks. Past performance does not guarantee future results. 
    Always do your own research and consult with qualified financial advisors before making investment decisions.
    This tool is for educational purposes only.
    """)

if __name__ == "__main__":
    main()
