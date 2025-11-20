import streamlit as st
import json
import os
import hashlib
from datetime import datetime
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Homepage", page_icon="🏠", layout="wide")

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
    .portfolio-card {
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Data directories
DATA_DIR = "data"
USERS_FILE = os.path.join(DATA_DIR, "users.json")
PORTFOLIOS_FILE = os.path.join(DATA_DIR, "portfolios.json")

def create_data_dir():
    """Create data directory if it doesn't exist"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    # Initialize files if they don't exist
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w') as f:
            json.dump({}, f)
    
    if not os.path.exists(PORTFOLIOS_FILE):
        with open(PORTFOLIOS_FILE, 'w') as f:
            json.dump({}, f)

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load users from JSON file"""
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_users(users):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def load_portfolios():
    """Load portfolios from JSON file"""
    try:
        with open(PORTFOLIOS_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_portfolios(portfolios):
    """Save portfolios to JSON file"""
    with open(PORTFOLIOS_FILE, 'w') as f:
        json.dump(portfolios, f, indent=2)

def register_user(username, password, email):
    """Register a new user"""
    users = load_users()
    if username in users:
        return False, "Username already exists"
    
    users[username] = {
        "password": hash_password(password),
        "email": email,
        "created_date": datetime.now().isoformat(),
        "total_investment": 0,
        "portfolio_value": 0
    }
    save_users(users)
    
    # Initialize empty portfolio
    portfolios = load_portfolios()
    portfolios[username] = {"stocks": {}, "transactions": []}
    save_portfolios(portfolios)
    
    return True, "User registered successfully"

def authenticate_user(username, password):
    """Authenticate user login"""
    users = load_users()
    if username not in users:
        return False, "Username not found"
    
    if users[username]["password"] != hash_password(password):
        return False, "Invalid password"
    
    return True, "Login successful"

def get_portfolio_value(username):
    """Calculate current portfolio value"""
    portfolios = load_portfolios()
    if username not in portfolios:
        return 0, {}
    
    portfolio = portfolios[username]["stocks"]
    total_value = 0
    stock_values = {}
    
    for symbol, quantity in portfolio.items():
        try:
            stock = yf.Ticker(symbol)
            current_price = stock.history(period="1d")['Close'].iloc[-1]
            stock_value = current_price * quantity
            total_value += stock_value
            stock_values[symbol] = {
                "quantity": quantity,
                "current_price": current_price,
                "value": stock_value
            }
        except:
            stock_values[symbol] = {
                "quantity": quantity,
                "current_price": 0,
                "value": 0
            }
    
    return total_value, stock_values

def add_to_portfolio(username, symbol, quantity, price):
    """Add stock to portfolio"""
    portfolios = load_portfolios()
    if username not in portfolios:
        portfolios[username] = {"stocks": {}, "transactions": []}
    
    # Update portfolio
    if symbol in portfolios[username]["stocks"]:
        portfolios[username]["stocks"][symbol] += quantity
    else:
        portfolios[username]["stocks"][symbol] = quantity
    
    # Add transaction
    transaction = {
        "symbol": symbol,
        "quantity": quantity,
        "price": price,
        "total": quantity * price,
        "type": "BUY",
        "date": datetime.now().isoformat()
    }
    portfolios[username]["transactions"].append(transaction)
    
    save_portfolios(portfolios)

def create_portfolio_chart(stock_values):
    """Create portfolio allocation chart"""
    if not stock_values:
        return None
    
    symbols = list(stock_values.keys())
    values = [stock_values[symbol]["value"] for symbol in symbols]
    
    fig = go.Figure(data=[go.Pie(labels=symbols, values=values, hole=0.3)])
    fig.update_layout(
        title="Portfolio Allocation",
        height=400,
        showlegend=True
    )
    return fig

def init_session_state():
    """Initialize session state"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None

def main():
    create_data_dir()
    init_session_state()
    
    st.markdown('<h1 class="main-header">🏠 Homepage & Dashboard</h1>', unsafe_allow_html=True)
    
    if not st.session_state.authenticated:
        # Login/Register tabs
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
        
        with tab1:
            st.subheader("Login to Your Account")
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                login_button = st.form_submit_button("Login", type="primary")
                
                if login_button:
                    if username and password:
                        success, message = authenticate_user(username, password)
                        if success:
                            st.session_state.authenticated = True
                            st.session_state.username = username
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        st.error("Please fill in all fields")
        
        with tab2:
            st.subheader("Create New Account")
            with st.form("register_form"):
                new_username = st.text_input("Choose Username")
                new_email = st.text_input("Email Address")
                new_password = st.text_input("Choose Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                register_button = st.form_submit_button("Register", type="primary")
                
                if register_button:
                    if all([new_username, new_email, new_password, confirm_password]):
                        if new_password != confirm_password:
                            st.error("Passwords do not match")
                        elif len(new_password) < 6:
                            st.error("Password must be at least 6 characters")
                        else:
                            success, message = register_user(new_username, new_password, new_email)
                            if success:
                                st.success(message)
                                st.info("Please login with your new account")
                            else:
                                st.error(message)
                    else:
                        st.error("Please fill in all fields")
    
    else:
        # Dashboard for authenticated users
        st.success(f"Welcome back, **{st.session_state.username}**! 👋")
        
        # Logout button in sidebar
        if st.sidebar.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
        
        # Portfolio overview
        st.markdown("## 📊 Portfolio Overview")
        
        portfolio_value, stock_values = get_portfolio_value(st.session_state.username)
        
        # Portfolio metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Portfolio Value", f"${portfolio_value:,.2f}")
        with col2:
            st.metric("Holdings", len(stock_values))
        with col3:
            # Calculate day change (simplified)
            st.metric("Today's Change", "$0.00", "0.00%")
        with col4:
            st.metric("Total Return", "$0.00", "0.00%")
        
        # Portfolio composition
        if stock_values:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Portfolio chart
                portfolio_fig = create_portfolio_chart(stock_values)
                if portfolio_fig:
                    st.plotly_chart(portfolio_fig, use_container_width=True)
            
            with col2:
                st.subheader("Holdings Details")
                for symbol, data in stock_values.items():
                    st.markdown(f"""
                    <div class="portfolio-card">
                        <h5>{symbol}</h5>
                        <p><strong>Quantity:</strong> {data['quantity']}</p>
                        <p><strong>Price:</strong> ${data['current_price']:.2f}</p>
                        <p><strong>Value:</strong> ${data['value']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("📈 Your portfolio is empty. Start by exploring our recommendation engines!")
        
        # Add to portfolio section
        st.markdown("## ➕ Add to Portfolio")
        with st.expander("Add New Stock"):
            col1, col2, col3 = st.columns(3)
            with col1:
                add_symbol = st.text_input("Stock Symbol (e.g., AAPL)")
            with col2:
                add_quantity = st.number_input("Quantity", min_value=1, value=1)
            with col3:
                add_price = st.number_input("Price per Share", min_value=0.01, value=100.0)
            
            if st.button("Add to Portfolio"):
                if add_symbol:
                    add_to_portfolio(st.session_state.username, add_symbol.upper(), add_quantity, add_price)
                    st.success(f"Added {add_quantity} shares of {add_symbol.upper()} to portfolio!")
                    st.rerun()
                else:
                    st.error("Please enter a stock symbol")
        
        # Recent transactions
        portfolios = load_portfolios()
        if st.session_state.username in portfolios:
            transactions = portfolios[st.session_state.username].get("transactions", [])
            if transactions:
                st.markdown("## 📈 Recent Transactions")
                df_transactions = pd.DataFrame(transactions[-10:])  # Last 10 transactions
                df_transactions['date'] = pd.to_datetime(df_transactions['date']).dt.strftime('%Y-%m-%d %H:%M')
                st.dataframe(df_transactions, use_container_width=True)

if __name__ == "__main__":
    main()
