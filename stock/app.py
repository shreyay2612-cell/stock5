

import streamlit as st
import json
import os
from datetime import datetime
import pandas as pd
import hashlib

# Configure Streamlit page
st.set_page_config(
    page_title="STOCK-5 | AI Stock Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .welcome-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

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
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📈 STOCK-5</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #666;">AI-Powered Stock Trading Platform</h3>', unsafe_allow_html=True)
    
    # Welcome section
    st.markdown("""
    <div class="welcome-card">
        <h2>🚀 Welcome to STOCK-5</h2>
        <p>Your comprehensive AI-powered stock analysis and recommendation platform</p>
        <p>Navigate to different pages using the sidebar to explore our features!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features overview
    st.markdown("## 🌟 Platform Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🏠 Homepage & Portfolio</h4>
            <p>Login to your account and manage your investment portfolio with real-time tracking</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>🎯 YFinance Recommendations</h4>
            <p>Get stock recommendations based on real-time market data and technical indicators</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Live Market Data</h4>
            <p>Access real-time stock prices, top performers, and comprehensive market analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>🤖 AI ML Predictions</h4>
            <p>Advanced LSTM-based stock predictions with confidence scores and risk analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Getting started
    st.markdown("## 🚀 Getting Started")
    st.info("""
    1. **Login**: Go to the Homepage to create an account or login
    2. **Explore Market**: Check live market data and top performers
    3. **Get Recommendations**: Use our AI-powered recommendation engines
    4. **Manage Portfolio**: Track your investments and performance
    """)
    
    # Authentication status
    if st.session_state.authenticated:
        st.success(f"✅ Logged in as: **{st.session_state.username}**")
    else:
        st.warning("🔐 Please login from the Homepage to access all features")

if __name__ == "__main__":
    main()
