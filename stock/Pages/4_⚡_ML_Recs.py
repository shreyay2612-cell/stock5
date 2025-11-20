import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import pickle
import warnings
import time
import json
import os
from datetime import datetime
import hashlib

warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="STOCK-5 | AI-Powered Stock Recommendations",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
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
    .risk-low { color: #28a745; font-weight: bold; }
    .risk-medium { color: #ffc107; font-weight: bold; }
    .risk-high { color: #fd7e14; font-weight: bold; }
    .risk-very-high { color: #dc3545; font-weight: bold; }
    .risk-extreme { color: #6f42c1; font-weight: bold; }
    .sidebar-info {
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the processed data"""
    try:
        train_df = pd.read_csv(r'C:\Users\shrey\Downloads\Stock Trade Recommendation-20250917T061356Z-1-001\Stock Trade Recommendation\Stock Trade Recommendation\Stock Trade Recommendation\train_processed.csv')
        test_df = pd.read_csv(r'C:\Users\shrey\Downloads\Stock Trade Recommendation-20250917T061356Z-1-001\Stock Trade Recommendation\Stock Trade Recommendation\Stock Trade Recommendation\test_processed.csv')
        return train_df, test_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

@st.cache_resource
def load_model_and_scalers():
    """Load and cache the model and scalers"""
    # Feature columns
    feature_columns = [
        'close', 'volume', 'macd', 'rsi_30', 'cci_30', 'dx_30', 
        'close_30_sma', 'close_60_sma', 'vix', 'turbulence',
        'llm_sentiment', 'llm_risk', 'price_change', 'volume_change',
        'high_low_ratio', 'close_open_ratio', 'boll_ub', 'boll_lb'
    ]
    
    # Rebuild model architecture
    def build_lstm_model(sequence_length=30, n_features=18, lstm_units=100):
        model = Sequential([
            LSTM(lstm_units, return_sequences=True, input_shape=(sequence_length, n_features)),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(lstm_units//2, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(lstm_units//4, return_sequences=False),
            Dropout(0.2),
            Dense(50, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    try:
        # Load scalers
        with open('stock_scalers1.pkl', 'rb') as f:
            scalers = pickle.load(f)
        
        # Build and load model
        model = build_lstm_model()
        model.load_weights('stock_lstm_model1.h5')
        
        return model, scalers, feature_columns
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

class StreamlitStockRecommendationEngine:
    def __init__(self, model, scalers, feature_columns, sequence_length=30):
        self.model = model
        self.scalers = scalers
        self.feature_columns = feature_columns
        self.sequence_length = sequence_length
        
    def prepare_prediction_data(self, df, stock_ticker):
        """Prepare data for prediction"""
        stock_data = df[df['tic'] == stock_ticker].copy()
        stock_data = stock_data.sort_values('date').reset_index(drop=True)
        
        if len(stock_data) < self.sequence_length:
            return None, None
            
        recent_data = stock_data.tail(self.sequence_length).copy()
        
        # Handle missing values
        for col in self.feature_columns:
            if col in recent_data.columns:
                recent_data.loc[:, col] = recent_data[col].replace([np.inf, -np.inf], np.nan)
                recent_data.loc[:, col] = recent_data[col].fillna(recent_data[col].median())
        
        # Get scaler
        if stock_ticker not in self.scalers:
            return None, None
        
        scaler = self.scalers[stock_ticker]
        scaled_features = scaler.transform(recent_data[self.feature_columns])
        X = scaled_features.reshape(1, self.sequence_length, len(self.feature_columns))
        
        return X, recent_data
    
    def calculate_confidence_score(self, prediction, recent_data):
        """Calculate confidence score"""
        confidence_factors = []
        
        # RSI-based confidence
        rsi = recent_data['rsi_30'].iloc[-1]
        if 30 <= rsi <= 70:
            rsi_confidence = 0.8
        elif 20 <= rsi <= 80:
            rsi_confidence = 0.6
        else:
            rsi_confidence = 0.4
        confidence_factors.append(rsi_confidence)
        
        # Volume consistency
        volume_std = recent_data['volume'].std() / (recent_data['volume'].mean() + 1e-8)
        volume_confidence = max(0.3, 0.9 - min(volume_std, 0.6))
        confidence_factors.append(volume_confidence)
        
        # Prediction magnitude
        pred_magnitude = abs(prediction)
        if pred_magnitude < 0.02:
            magnitude_confidence = 0.8
        elif pred_magnitude < 0.05:
            magnitude_confidence = 0.6
        else:
            magnitude_confidence = 0.4
        confidence_factors.append(magnitude_confidence)
        
        # VIX-based confidence
        vix = recent_data['vix'].iloc[-1]
        if vix < 20:
            vix_confidence = 0.8
        elif vix < 30:
            vix_confidence = 0.6
        else:
            vix_confidence = 0.4
        confidence_factors.append(vix_confidence)
        
        return min(0.95, max(0.15, np.mean(confidence_factors)))
    
    def calculate_risk_factor(self, recent_data, prediction):
        """Calculate risk factor"""
        risk_factors = []
        
        # Historical volatility
        volatility = recent_data['price_change'].std()
        if pd.isna(volatility):
            volatility = 0.03
            
        if volatility < 0.02:
            risk_factors.append(1)
        elif volatility < 0.04:
            risk_factors.append(2)
        elif volatility < 0.06:
            risk_factors.append(3)
        elif volatility < 0.08:
            risk_factors.append(4)
        else:
            risk_factors.append(5)
        
        # VIX-based risk
        vix = recent_data['vix'].iloc[-1]
        if vix < 15:
            risk_factors.append(1)
        elif vix < 25:
            risk_factors.append(2)
        elif vix < 35:
            risk_factors.append(3)
        elif vix < 45:
            risk_factors.append(4)
        else:
            risk_factors.append(5)
        
        # Prediction magnitude risk
        pred_magnitude = abs(prediction)
        if pred_magnitude > 0.05:
            risk_factors.append(4)
        elif pred_magnitude > 0.03:
            risk_factors.append(3)
        else:
            risk_factors.append(2)
        
        return min(5, max(1, round(np.mean(risk_factors))))
    
    def get_stock_recommendations(self, df, user_investment, top_n=5, progress_bar=None):
        """Generate stock recommendations with progress tracking"""
        recommendations = []
        available_stocks = [stock for stock in df['tic'].unique() if stock in self.scalers]
        
        for i, stock in enumerate(available_stocks):
            if progress_bar:
                progress_bar.progress((i + 1) / len(available_stocks))
            
            try:
                result = self.prepare_prediction_data(df, stock)
                if result[0] is None:
                    continue
                    
                X, recent_data = result
                prediction = self.model.predict(X, verbose=0)[0][0]
                current_price = recent_data['close'].iloc[-1]
                
                if current_price <= 0:
                    continue
                
                max_shares = int(user_investment / current_price)
                if max_shares == 0:
                    continue
                
                confidence = self.calculate_confidence_score(prediction, recent_data)
                risk_level = self.calculate_risk_factor(recent_data, prediction)
                
                potential_profit = max_shares * current_price * prediction
                composite_score = prediction * confidence * (6 - risk_level) * 100
                
                recommendations.append({
                    'stock': stock,
                    'current_price': current_price,
                    'predicted_return': prediction,
                    'confidence_score': confidence,
                    'risk_level': risk_level,
                    'max_shares': max_shares,
                    'investment_amount': max_shares * current_price,
                    'potential_profit': potential_profit,
                    'composite_score': composite_score,
                    'rsi': recent_data['rsi_30'].iloc[-1],
                    'volume': recent_data['volume'].iloc[-1],
                    'sentiment': recent_data['llm_sentiment'].iloc[-1] if not pd.isna(recent_data['llm_sentiment'].iloc[-1]) else None
                })
                
            except Exception:
                continue
        
        recommendations.sort(key=lambda x: x['composite_score'], reverse=True)
        return recommendations[:top_n]

def create_portfolio_charts(recommendations):
    """Create interactive portfolio visualization charts"""
    if not recommendations:
        return None
    
    stocks = [rec['stock'] for rec in recommendations]
    returns = [rec['predicted_return'] * 100 for rec in recommendations]
    confidence = [rec['confidence_score'] * 100 for rec in recommendations]
    risks = [rec['risk_level'] for rec in recommendations]
    profits = [rec['potential_profit'] for rec in recommendations]
    investments = [rec['investment_amount'] for rec in recommendations]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Predicted Returns (%)', 'Risk vs Confidence', 
                       'Potential Profits ($)', 'Investment Allocation'),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "pie"}]]
    )
    
    # Returns bar chart
    colors = ['green' if r > 0 else 'red' for r in returns]
    fig.add_trace(
        go.Bar(x=stocks, y=returns, marker_color=colors, name="Returns", showlegend=False),
        row=1, col=1
    )
    
    # Risk vs Confidence scatter
    fig.add_trace(
        go.Scatter(
            x=confidence, y=risks, mode='markers+text',
            marker=dict(size=[abs(p)/100 for p in profits], color=returns, 
                       colorscale='RdYlGn', showscale=True),
            text=stocks, textposition="top center",
            name="Risk vs Confidence", showlegend=False
        ),
        row=1, col=2
    )
    
    # Profits bar chart
    profit_colors = ['green' if p > 0 else 'red' for p in profits]
    fig.add_trace(
        go.Bar(x=stocks, y=profits, marker_color=profit_colors, name="Profits", showlegend=False),
        row=2, col=1
    )
    
    # Investment allocation pie chart
    fig.add_trace(
        go.Pie(labels=stocks, values=investments, name="Allocation", showlegend=False),
        row=2, col=2
    )
    
    fig.update_layout(height=600, title_text="Portfolio Analysis Dashboard")
    return fig

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
    
    # Your existing ML recommendation code here
    # Just replace the original main() function content
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI ML Stock Predictions</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #666;">LSTM Deep Learning Model Recommendations</h3>', unsafe_allow_html=True)
    
    # Load data and model
    with st.spinner("Loading AI models and market data..."):
        train_df, test_df = load_data()
        model, scalers, feature_columns = load_model_and_scalers()
    
    if not all([train_df is not None, test_df is not None, model is not None]):
        st.error("Failed to load required data and models. Please ensure all files are available.")
        return
    
    # Sidebar
    st.sidebar.markdown("## 💰 Investment Settings")
    
    # User input for investment amount
    investment_amount = st.sidebar.number_input(
        "Investment Amount ($)",
        min_value=1000,
        max_value=1000000,
        value=10000,
        step=1000,
        help="Enter the amount you want to invest"
    )
    
    # Number of recommendations
    num_recommendations = st.sidebar.slider(
        "Number of Recommendations",
        min_value=3,
        max_value=10,
        value=5,
        help="How many stock recommendations to generate"
    )
    
    # Analysis settings
    st.sidebar.markdown("## ⚙️ Analysis Settings")
    show_details = st.sidebar.checkbox("Show detailed analysis", value=True)
    show_charts = st.sidebar.checkbox("Show portfolio charts", value=True)
    
    # Market info sidebar
    st.sidebar.markdown("## 📊 Market Info")
    if test_df is not None:
        total_stocks = len(test_df['tic'].unique())
        latest_date = test_df['date'].max()
        avg_vix = test_df['vix'].iloc[-100:].mean()
        
        st.sidebar.markdown(f"""
        <div class="sidebar-info">
        <strong>Available Stocks:</strong> {total_stocks}<br>
        <strong>Latest Data:</strong> {latest_date}<br>
        <strong>Market Volatility (VIX):</strong> {avg_vix:.1f}
        </div>
        """, unsafe_allow_html=True)
    
    # Generate recommendations button
    if st.sidebar.button("🎯 Generate Recommendations", type="primary"):
        
        # Initialize recommendation engine
        recommender = StreamlitStockRecommendationEngine(model, scalers, feature_columns)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Analyzing market data and generating predictions...")
        
        # Generate recommendations
        recommendations = recommender.get_stock_recommendations(
            test_df, investment_amount, num_recommendations, progress_bar
        )
        
        progress_bar.empty()
        status_text.empty()
        
        if not recommendations:
            st.error("No recommendations could be generated. Please try a different investment amount.")
            return
        
        # Display recommendations
        st.markdown("## 🎯 Top Stock Recommendations")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_investment = sum(rec['investment_amount'] for rec in recommendations)
        total_potential = sum(rec['potential_profit'] for rec in recommendations)
        avg_confidence = np.mean([rec['confidence_score'] for rec in recommendations])
        avg_risk = np.mean([rec['risk_level'] for rec in recommendations])
        
        with col1:
            st.metric("Total Investment", f"${total_investment:,.2f}")
        with col2:
            st.metric("Potential Profit", f"${total_potential:,.2f}")
        with col3:
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        with col4:
            st.metric("Avg Risk Level", f"{avg_risk:.1f}/5.0")
        
        # Detailed recommendations
        for i, rec in enumerate(recommendations, 1):
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])
                
                # Risk color mapping
                risk_colors = {1: "risk-low", 2: "risk-medium", 3: "risk-high", 4: "risk-very-high", 5: "risk-extreme"}
                risk_labels = {1: "🟢 Low", 2: "🟡 Medium", 3: "🟠 High", 4: "🔴 Very High", 5: "🚨 Extreme"}
                
                with col1:
                    st.markdown(f"### #{i}. {rec['stock']}")
                    st.write(f"**Current Price:** ${rec['current_price']:.2f}")
                    st.write(f"**Max Shares:** {rec['max_shares']:,}")
                    st.write(f"**Investment:** ${rec['investment_amount']:,.2f}")
                
                with col2:
                    st.metric("Predicted Return", f"{rec['predicted_return']:.2%}")
                    st.metric("Confidence Score", f"{rec['confidence_score']:.1%}")
                
                with col3:
                    st.metric("Potential Profit", f"${rec['potential_profit']:,.2f}")
                    st.markdown(f"**Risk Level:** <span class='{risk_colors[rec['risk_level']]}'>{risk_labels[rec['risk_level']]}</span>", unsafe_allow_html=True)
                
                if show_details:
                    with st.expander(f"📊 Detailed Analysis - {rec['stock']}"):
                        detail_col1, detail_col2 = st.columns(2)
                        with detail_col1:
                            st.write(f"**RSI:** {rec['rsi']:.1f}")
                            st.write(f"**Volume:** {rec['volume']:,.0f}")
                        with detail_col2:
                            if rec['sentiment']:
                                st.write(f"**Sentiment:** {rec['sentiment']:.1f}/5.0")
                            st.write(f"**Composite Score:** {rec['composite_score']:.2f}")
                
                st.divider()
        
        # Portfolio charts
        if show_charts:
            st.markdown("## 📈 Portfolio Analysis")
            portfolio_fig = create_portfolio_charts(recommendations)
            if portfolio_fig:
                st.plotly_chart(portfolio_fig, use_container_width=True)
        
        # Risk disclaimer
        st.markdown("## ⚠️ Important Disclaimer")
        st.warning("""
        **Investment Risk Warning:** This tool provides AI-generated stock predictions for educational and research purposes only. 
        Stock market investments carry inherent risks, and past performance does not guarantee future results. 
        Always consult with qualified financial advisors before making investment decisions. 
        The creators are not responsible for any financial losses resulting from using these recommendations.
        """)
main()
# Footer
st.markdown("---")
st.markdown("**STOCK-5** | Deep Learning Based Stock Trading Recommendation Engine | Built with Streamlit")
