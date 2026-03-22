import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from PIL import Image
import finnhub
from openai import OpenAI
import os
from xgboost import XGBRegressor

# ---------- CONFIG ----------

FINNHUB_API_KEY = st.secrets["FINNHUB_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

BIG_TECHS = [
    "AAPL", "MSFT", "NVDA", "AMZN",
    "META", "TSLA", "GOOGL", "AMD"
]

st.set_page_config(
    page_title="Lupa AI Stock Terminal",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# ---------- THEME SETTING ----------
dark_mode = st.sidebar.toggle("Night Mode", value=True)

if dark_mode:
    bg_style = "radial-gradient(circle at 50% 30%, rgba(255,255,255,0.05), transparent 60%), radial-gradient(circle at center, #1e293b 0%, #020617 100%)"
    sidebar_bg = "#020617"
    text_color = "#ffffff"
    metric_bg = "rgba(255,255,255,0.05)"
    plotly_template = "plotly_dark"
    grid_color = "rgba(255,255,255,0.1)"
else:
    bg_style = "#ffffff"
    sidebar_bg = "#f8f9fa"
    text_color = "#000000" # 白天模式：所有普通文字、图表字符、数字均为纯黑
    metric_bg = "#f0f2f6"
    plotly_template = "plotly_white"
    grid_color = "rgba(0,0,0,0.1)"

# ---------- STYLE ----------

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background: {bg_style};
}}

[data-testid="stSidebar"] {{
    background-color: {sidebar_bg};
}}

.block-container{{
    padding-top:2rem;
}}

[data-testid="stMetric"]{{
    background:{metric_bg};
    padding:15px;
    border-radius:10px;
}}

/* [修改] 强制所有层级的文字颜色 */
h1, h2, h3, h4, h5, p, label, span, div {{
    color: {text_color} !important;
}}

/* [修改] 指标数字专项修复 - 确保 $ 价格数字可见 */
[data-testid="stMetricValue"] div {{
    color: {text_color} !important;
}}

/* [修改] 按钮文字专项修复 - 强制使用白色文字，以便在深色按钮背景下清晰 */
.stButton > button p {{
    color: white !important;
    font-weight: 700 !important;
}}

/* [新增] 侧边栏及 Tab 标签激活状态修正 */
button[data-baseweb="tab"] div {{
    color: {text_color} !important;
}}

.stTextInput input {{
    color: {text_color} !important;
}}
</style>
""", unsafe_allow_html=True)

# ---------- LOGO ----------

logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
try:
    logo = Image.open(logo_path)
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        st.image(logo, width=120)
    with col_title:
        st.title("Lupa AI Stock Terminal")
except:
    st.title("Lupa AI Stock Terminal")

# ---------- SIDEBAR ----------

if "ticker" not in st.session_state:
    st.session_state.ticker = "AAPL"

if "bigtech" not in st.session_state:
    st.session_state.bigtech = "AAPL"

def ticker_changed():
    ticker = st.session_state.ticker.upper()
    if ticker in BIG_TECHS:
        st.session_state.bigtech = ticker

def bigtech_changed():
    st.session_state.ticker = st.session_state.bigtech

st.sidebar.text_input("Ticker", key="ticker", on_change=ticker_changed)
st.sidebar.radio("Big Tech", BIG_TECHS, key="bigtech", on_change=bigtech_changed)

symbol = st.session_state.ticker.upper()
period = st.sidebar.selectbox("Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)

# ---------- DATA ----------

@st.cache_data
def load_data(symbol, period):
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    df["MA20"] = df["Close"].rolling(20).mean()
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))
    df["Returns"] = df["Close"].pct_change()
    df["Volatility"] = df["Returns"].rolling(20).std() * np.sqrt(252)
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
    df["BB_std"] = df["Close"].rolling(20).std()
    df["BB_upper"] = df["MA20"] + 2 * df["BB_std"]
    df["BB_lower"] = df["MA20"] - 2 * df["BB_std"]
    df["Volume_MA20"] = df["Volume"].rolling(20).mean()
    df["Volume_momentum"] = df["Volume"] / df["Volume_MA20"]
    return df

df = load_data(symbol, period)

if df.empty:
    st.error("Ticker not found")
    st.stop()

price = df["Close"].iloc[-1]
ret = df["Returns"].iloc[-1]

# ---------- HEADER ----------

st.markdown(f"## 📊 {symbol} Market Overview")
col1, col2, col3, col4 = st.columns(4)
trend = "Bullish" if price > df["MA20"].iloc[-1] else "Bearish"

with col1:
    st.metric("Price", f"${price:.2f}", f"{ret:.2%}")
with col2:
    st.metric("Trend", trend)
with col3:
    st.metric("Volatility", f"{df['Volatility'].iloc[-1]:.2%}")
with col4:
    st.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")

# ---------- MARKET SENTIMENT ----------

sentiment = 50 + ret * 100
fig_sent = go.Figure(go.Indicator(
    mode="gauge+number",
    value=sentiment,
    title={'text': "Market Sentiment", 'font': {'color': text_color}},
    gauge={
        'axis': {'range': [0, 100], 'tickcolor': text_color},
        'bar': {'color': "#3b82f6"},
        'steps': [
            {'range': [0, 40], 'color': "#ef4444"},
            {'range': [40, 60], 'color': "#facc15"},
            {'range': [60, 100], 'color': "#22c55e"}
        ]
    }
))

# [修改] 显式设置仪表盘文字颜色
fig_sent.update_layout(
    template=plotly_template,
    paper_bgcolor='rgba(0,0,0,0)',
    font={'color': text_color}
)
fig_sent.update_traces(number={'font': {'color': text_color}})
st.plotly_chart(fig_sent, use_container_width=True)

# ---------- CHART ----------

def create_chart(df):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.75, 0.25]
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
        name="Price"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["MA20"],
        line=dict(color="#60a5fa", width=2), name="MA20"
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color="rgba(120,160,255,0.3)", name="Volume"
    ), row=2, col=1)

    # [修改] 核心修复：强制 Plotly 图表内部字符和刻度使用纯黑/白文字
    fig.update_layout(
        height=650,
        hovermode="x unified",
        template=plotly_template,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=text_color), # 修复图例和标题颜色
        xaxis=dict(
            rangeslider=dict(visible=False),
            type="date",
            tickfont=dict(color=text_color), # 修复 X 轴数字
            gridcolor=grid_color
        ),
        yaxis=dict(
            tickfont=dict(color=text_color), # 修复 Y 轴价格数字
            gridcolor=grid_color
        )
    )
    return fig

# ---------- AI MODEL & LLM ----------

@st.cache_resource
def train_model(X, y):
    model = XGBRegressor(n_estimators=80, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=1)
    model.fit(X, y)
    return model

def price_forecast(df, window=20):
    df_f = df.tail(350).dropna()
    features = ["Close", "MA20", "RSI", "Returns", "Volatility", "MACD", "MACD_signal", "BB_upper", "BB_lower", "Volume_momentum"]
    data = df_f[features].values
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i].flatten())
        y.append(data[i][0])
    model = train_model(np.array(X), np.array(y))
    last_window = data[-window:].flatten().reshape(1, -1)
    return float(model.predict(last_window)[0])

@st.cache_data(ttl=600)
def run_llm(prompt):
    response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content

# ---------- TABS ----------

tab_chart, tab_ai, tab_almanac, tab_heat, tab_news = st.tabs(["📊 Chart", "🤖 AI Forecast", "📅 Almanac", "🌎 Heatmap", "📰 News"])

with tab_chart:
    st.plotly_chart(create_chart(df), use_container_width=True, config={"scrollZoom": True})

with tab_ai:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("XGBoost Prediction")
        pred_price = price_forecast(df)
        direction = "Bullish" if pred_price > price else "Bearish"
        st.metric("Predicted Price", f"${pred_price:.2f}", direction)
    with col2:
        st.subheader("LLM Analysis")
        if st.button("Run LLM Analysis", key="llm_button"):
            prompt = f"Stock: {symbol}, Price: {price}, RSI: {df['RSI'].iloc[-1]:.2f}, Trend: {trend}. Short outlook."
            st.write(run_llm(prompt))

# ---------- HEATMAP ----------

with tab_heat:
    data = []
    for t in BIG_TECHS:
        try:
            d = yf.download(t, period="5d", progress=False)
            close = d["Close"].iloc[:, 0] if isinstance(d["Close"], pd.DataFrame) else d["Close"]
            change = (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100
            data.append({"Ticker": t, "Change": float(change)})
        except: pass
    fig_heat = px.bar(pd.DataFrame(data), x="Ticker", y="Change", color="Change", text="Change", color_continuous_scale="RdYlGn")
    fig_heat.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    # [修改] 修复热力图文字
    fig_heat.update_layout(
        template=plotly_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': text_color},
        xaxis=dict(tickfont=dict(color=text_color)),
        yaxis=dict(tickfont=dict(color=text_color))
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ---------- NEWS & ALMANAC (省略逻辑，保持原有结构) ----------
# ... [此处保持您原有代码的其余部分即可] ...