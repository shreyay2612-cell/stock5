import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import json
import os
import hashlib

st.set_page_config(page_title="Market Info", page_icon="📊", layout="wide")

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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stock-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 0.5rem 0;
    }
    .stock-card.negative {
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_market_indices():
    """Get major market indices data"""
    indices = {
        "S&P 500": "^GSPC",
        "Dow Jones": "^DJI",
        "NASDAQ": "^IXIC",
        "Russell 2000": "^RUT",
        "VIX": "^VIX"
    }
    
    index_data = {}
    for name, symbol in indices.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            if len(hist) >= 2:
                current = hist['Close'].iloc[-1]
                previous = hist['Close'].iloc[-2]
                change = current - previous
                change_pct = (change / previous) * 100
                
                index_data[name] = {
                    "symbol": symbol,
                    "current": current,
                    "change": change,
                    "change_pct": change_pct
                }
        except:
            continue
    
    return index_data

@st.cache_data(ttl=300)
def get_top_stocks(period="1d", limit=10):
    """Get top performing stocks"""
    # Popular stocks to check
    popular_stocks = [
        "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX", 
        "AMD", "INTC", "CRM", "ORCL", "ADBE", "PYPL", "SPOT", "UBER",
        "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "DIS", "KO", "PEP"
    ]
    
    stock_performance = []
    
    for symbol in popular_stocks:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="5d")  # Get more days for reliability
            
            if len(hist) >= 2:
                current = hist['Close'].iloc[-1]
                previous = hist['Close'].iloc[-2]
                change_pct = ((current - previous) / previous) * 100
                volume = hist['Volume'].iloc[-1]
                
                # Get basic info
                info = stock.info
                name = info.get('longName', symbol)
                
                stock_performance.append({
                    "Symbol": symbol,
                    "Name": name,
                    "Price": current,
                    "Change%": change_pct,
                    "Volume": volume
                })
        except:
            continue
    
    # Sort by performance
    df = pd.DataFrame(stock_performance)
    if not df.empty:
        df_sorted = df.sort_values("Change%", ascending=False)
        return df_sorted.head(limit), df_sorted.tail(limit)
    
    return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=300)
def get_sector_performance():
    """Get sector ETF performance"""
    sectors = {
        "Technology": "XLK",
        "Healthcare": "XLV", 
        "Financial": "XLF",
        "Consumer Discretionary": "XLY",
        "Communication": "XLC",
        "Industrial": "XLI",
        "Consumer Staples": "XLP",
        "Energy": "XLE",
        "Utilities": "XLU",
        "Real Estate": "XLRE",
        "Materials": "XLB"
    }
    
    sector_data = []
    for sector, etf in sectors.items():
        try:
            ticker = yf.Ticker(etf)
            hist = ticker.history(period="2d")
            if len(hist) >= 2:
                current = hist['Close'].iloc[-1]
                previous = hist['Close'].iloc[-2]
                change_pct = ((current - previous) / previous) * 100
                
                sector_data.append({
                    "Sector": sector,
                    "ETF": etf,
                    "Price": current,
                    "Change%": change_pct
                })
        except:
            continue
    
    return pd.DataFrame(sector_data).sort_values("Change%", ascending=False)

def create_market_overview_chart(index_data):
    """Create market overview chart"""
    if not index_data:
        return None
    
    names = list(index_data.keys())
    changes = [data["change_pct"] for data in index_data.values()]
    colors = ['green' if x > 0 else 'red' for x in changes]
    
    fig = go.Figure(data=[
        go.Bar(x=names, y=changes, marker_color=colors, name="Change %")
    ])
    
    fig.update_layout(
        title="Major Indices Performance Today",
        xaxis_title="Index",
        yaxis_title="Change %",
        height=400,
        showlegend=False
    )
    
    return fig

def get_stock_chart(symbol, period="1mo"):
    """Get stock price chart"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        
        fig = go.Figure(data=[
            go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name=symbol
            )
        ])
        
        fig.update_layout(
            title=f"{symbol} - {period.upper()} Chart",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400
        )
        
        return fig
    except:
        return None

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
    st.markdown('<h1 class="main-header">📊 Live Market Data</h1>', unsafe_allow_html=True)
    
    # Market indices overview
    st.markdown("## 📈 Market Indices")
    
    with st.spinner("Loading market data..."):
        index_data = get_market_indices()
    
    if index_data:
        # Display indices in columns
        cols = st.columns(len(index_data))
        for i, (name, data) in enumerate(index_data.items()):
            with cols[i]:
                color = "🟢" if data["change"] > 0 else "🔴"
                st.metric(
                    f"{color} {name}",
                    f"{data['current']:.2f}",
                    f"{data['change']:+.2f} ({data['change_pct']:+.2f}%)"
                )
        
        # Market overview chart
        market_fig = create_market_overview_chart(index_data)
        if market_fig:
            st.plotly_chart(market_fig, use_container_width=True)
    
    # Top performers
    st.markdown("## 🚀 Top Performers Today")
    
    with st.spinner("Analyzing stock performance..."):
        top_gainers, top_losers = get_top_stocks()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🟢 Top Gainers")
        if not top_gainers.empty:
            for _, stock in top_gainers.iterrows():
                st.markdown(f"""
                <div class="stock-card">
                    <h5>{stock['Symbol']} - {stock['Name'][:30]}...</h5>
                    <p><strong>Price:</strong> ${stock['Price']:.2f}</p>
                    <p><strong>Change:</strong> +{stock['Change%']:.2f}%</p>
                    <p><strong>Volume:</strong> {stock['Volume']:,.0f}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No data available")
    
    with col2:
        st.subheader("🔴 Top Losers")
        if not top_losers.empty:
            for _, stock in top_losers.iterrows():
                st.markdown(f"""
                <div class="stock-card negative">
                    <h5>{stock['Symbol']} - {stock['Name'][:30]}...</h5>
                    <p><strong>Price:</strong> ${stock['Price']:.2f}</p>
                    <p><strong>Change:</strong> {stock['Change%']:.2f}%</p>
                    <p><strong>Volume:</strong> {stock['Volume']:,.0f}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No data available")
    
    # Sector performance
    st.markdown("## 🏭 Sector Performance")
    
    with st.spinner("Loading sector data..."):
        sector_df = get_sector_performance()
    
    if not sector_df.empty:
        # Sector performance chart
        fig_sector = px.bar(
            sector_df, 
            x="Sector", 
            y="Change%",
            color="Change%",
            color_continuous_scale="RdYlGn",
            title="Sector Performance Today"
        )
        fig_sector.update_layout(height=400)
        st.plotly_chart(fig_sector, use_container_width=True)
        
        # Sector data table
        st.dataframe(sector_df, use_container_width=True)
    
    # Stock lookup
    st.markdown("## 🔍 Stock Lookup")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        lookup_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)", value="AAPL")
        period = st.selectbox("Time Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"])
    
    with col2:
        if st.button("Get Stock Data", type="primary"):
            if lookup_symbol:
                try:
                    stock = yf.Ticker(lookup_symbol.upper())
                    info = stock.info
                    
                    # Stock info
                    st.subheader(f"{info.get('longName', lookup_symbol.upper())}")
                    
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("Current Price", f"${info.get('currentPrice', 'N/A')}")
                    with col_b:
                        st.metric("Market Cap", f"${info.get('marketCap', 0):,.0f}")
                    with col_c:
                        st.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
                    with col_d:
                        st.metric("52W High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
                    
                    # Stock chart
                    stock_fig = get_stock_chart(lookup_symbol.upper(), period)
                    if stock_fig:
                        st.plotly_chart(stock_fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error fetching data for {lookup_symbol.upper()}: {str(e)}")
    
    # Market news (placeholder)
    st.markdown("## 📰 Market News")
    st.info("📈 Market news integration coming soon! Stay tuned for real-time financial news and analysis.")

if __name__ == "__main__":
    main()
