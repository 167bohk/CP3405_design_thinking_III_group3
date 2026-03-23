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
import json

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
    muted_text_color = "#ffffff"
    metric_bg = "rgba(255,255,255,0.05)"
    card_bg = "rgba(255,255,255,0.06)"
    card_border = "1px solid rgba(255,255,255,0.10)"
    plotly_template = "plotly_dark"
    grid_color = "rgba(255,255,255,0.1)"
else:
    bg_style = bg_style = "radial-gradient(circle at 50% 30%, rgba(0,0,0,0.12), transparent 55%), radial-gradient(circle at center, #ffffff 0%, #cbd5e1 100%)"
    sidebar_bg = "#ffffff"
    text_color = "#000000"
    muted_text_color = "#000000"
    metric_bg = "#ffffff"
    card_bg = "rgba(255,255,255,0.92)"
    card_border = "1px solid rgba(15,23,42,0.08)"
    plotly_template = "plotly_white"
    grid_color = "rgba(0,0,0,0.1)"

# ---------- STYLE ----------


st.markdown(f"""
<style>

[data-testid="stAppViewContainer"] {{
    background: {bg_style} !important;
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

h1, h2, h3, h4, h5, p, label, span, div {{
    color: {text_color};
}}

[data-testid="stSidebar"] *,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {{
    color: {text_color} !important;
}}

[data-testid="stMetricValue"] div {{
    color: {text_color} !important;
}}

.stButton > button p {{
    color: white !important;
    font-weight: 700 !important;
}}

button[data-baseweb="tab"] div {{
    color: {text_color} !important;
}}

.stTextInput input,
.stSelectbox div[data-baseweb="select"] > div,
.stSelectbox input {{
    color: {text_color} !important;
    -webkit-text-fill-color: {text_color} !important;
}}

.themed-card {{
    background: {card_bg};
    border: {card_border};
    border-radius: 15px;
}}

.signal-card {{
    background: {card_bg};
    border: {card_border};
    border-radius: 15px;
    text-align: center;
}}

.signal-card-title {{
    margin: 0;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.2;
}}

.signal-buy {{
    color: #22c55e !important;
}}

.signal-sell {{
    color: #ef4444 !important;
}}

</style>
""", unsafe_allow_html=True)


def get_signal_style(value, reference):
    if value > reference:
        return "↑ Bullish", "#22c55e"
    else:
        return "↓ Bearish", "#ef4444"
    
# ---------- LOGO ----------

logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
logo = Image.open(logo_path)

col_logo, col_title = st.columns([1, 4])

with col_logo:
    st.image(logo, width=120)

with col_title:
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
    for key in [
        "ensemble_price",
        "llm_price",
        "pred_price",
        "llm_reason",
        "llm_conf"
    ]:
        if key in st.session_state:
            del st.session_state[key]


def bigtech_changed():
    st.session_state.ticker = st.session_state.bigtech

    for key in [
        "ensemble_price",
        "llm_price",
        "pred_price",
        "llm_reason",
        "llm_conf"
    ]:
        if key in st.session_state:
            del st.session_state[key]


st.sidebar.text_input("Ticker", key="ticker", on_change=ticker_changed)

st.sidebar.radio("Big Tech", BIG_TECHS, key="bigtech", on_change=bigtech_changed)

symbol = st.session_state.ticker.upper()

period = st.sidebar.selectbox(
    "Period",
    ["3mo", "6mo", "1y", "2y", "5y"],
    index=2
)


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

    # MACD
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()

    # Bollinger Bands
    df["BB_std"] = df["Close"].rolling(20).std()
    df["BB_upper"] = df["MA20"] + 2 * df["BB_std"]
    df["BB_lower"] = df["MA20"] - 2 * df["BB_std"]

    # Volume momentum
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
        'axis': {'range': [0, 100], 'tickcolor': text_color, 'tickfont': {'color': text_color}},  
        'bar': {'color': "#3b82f6"},
        'steps': [
            {'range': [0, 40], 'color': "#ef4444"},
            {'range': [40, 60], 'color': "#facc15"},
            {'range': [60, 100], 'color': "#22c55e"}
        ]
    }
))

fig_sent.update_layout(template=plotly_template, paper_bgcolor='rgba(0,0,0,0)', font={'color': text_color})
fig_sent.update_traces(number={'font': {'color': text_color}})

st.plotly_chart(fig_sent, use_container_width=True)


# ---------- CHART ----------

def create_chart(df):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25]
    )

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        increasing_line_color="#22c55e",
        decreasing_line_color="#ef4444"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["MA20"],
        line=dict(color="#60a5fa", width=2),
        name="MA20"
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df.index,
        y=df["Volume"],
        marker_color="rgba(120,160,255,0.3)"
    ), row=2, col=1)

    fig.update_layout(
        height=650,
        hovermode="x unified",
        dragmode="pan"
    )

    fig.update_layout(
        template=plotly_template,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': text_color}
    )

    fig.update_xaxes(tickfont=dict(color=text_color), gridcolor=grid_color)  
    fig.update_yaxes(tickfont=dict(color=text_color), gridcolor=grid_color)  

    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date"
        )
    )

    return fig



# ---------- XGBOOST MODEL ----------

@st.cache_resource
def train_model(X, y):
    model = XGBRegressor(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=1
    )

    model.fit(X, y)

    return model


def price_forecast(df, window=20):
    df = df.tail(350)
    df = df.dropna()

    features = [
        "Close", "MA20", "RSI", "Returns", "Volatility",
        "MACD", "MACD_signal", "BB_upper", "BB_lower",
        "Volume_momentum"
    ]

    data = df[features].values

    X = []
    y = []

    for i in range(window, len(data)):
        X.append(data[i - window:i].flatten())
        y.append(data[i][0])

    X = np.array(X)
    y = np.array(y)

    model = train_model(X, y)

    last_window = data[-window:].flatten().reshape(1, -1)

    pred = model.predict(last_window)

    return float(pred[0])


# ---------- LLM ----------

@st.cache_data(ttl=600)
def run_llm(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    return response.choices[0].message.content



# ---------- TABS ----------

tab_chart, tab_ai, tab_almanac, tab_heat, tab_news = st.tabs([
    "📊 Chart",
    "🤖 AI Forecast",
    "📅 Almanac",
    "🌎 Heatmap",
    "📰 News"
])
# ---------- CHART ----------

with tab_chart:
    fig = create_chart(df)
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"scrollZoom": True}
    )
# ---------- NEWS DATA (GLOBAL) ----------

@st.cache_data(ttl=600)
def get_news(symbol):
    today = datetime.today().strftime("%Y-%m-%d")
    last_week = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")

    try:
        return finnhub_client.company_news(symbol, _from=last_week, to=today)
    except:
        return []

news = get_news(symbol)

# ---------- ALMANAC DATA (GLOBAL) ----------

spy = yf.download("SPY", period="2y", progress=False)
jan = spy[spy.index.month == 1]

# January Barometer
if len(jan) > 5:
    close = jan["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    jan_return = float((close.iloc[-1] / close.iloc[0]) - 1)
    jan_signal = "Bullish" if jan_return > 0 else "Bearish"
else:
    jan_signal = "Neutral"

# First Five Days
jan5 = jan.head(5)

if len(jan5) == 5:
    close = jan5["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    jan5_return = float((close.iloc[-1] / close.iloc[0]) - 1)
    five_signal = "Bullish" if jan5_return > 0 else "Bearish"
else:
    five_signal = "Neutral"

# Best Six Months

def best_six_months():
    month = datetime.now().month

    if month in [11, 12, 1, 2, 3, 4]:
        return "Bullish Season"
    else:
        return "Weak Season"
    
best6 = best_six_months()

# Presidential Cycle
 
year = datetime.now().year
cycle = year % 4

if cycle == 0:
    pres = "Election Year"
elif cycle == 1:
    pres = "Post Election"
elif cycle == 2:
    pres = "Midterm Weakness"
else:
    pres = "Pre Election Bullish"

# ---------- NEWS SUMMARY (FOR LLM) ----------

news_summary = " | ".join(
    [n.get("headline", "")[:120] for n in news[:5] if n.get("headline")]
)

if not news_summary:
    news_summary = "No significant recent news."
    
# ---------- LLM ----------



with tab_ai:
    col1, col2 = st.columns(2)

    with col1:

        st.subheader("XGBoost Prediction")

        pred_price = price_forecast(df)

        signal_text, signal_color = get_signal_style(pred_price, price)

        st.markdown(f"""
        <div style="
            background: {card_bg};
            border: {card_border};
            padding: 20px;
            border-radius: 15px;
        ">
            <p style="color:{muted_text_color};">Predicted Price</p>
            <h2>${pred_price:.2f}</h2>
            <span style="color:{signal_color}; font-weight:600;">
                {signal_text}
            </span>
        </div>
        """, unsafe_allow_html=True)

    with col2:



        prompt = f"""
        You are a professional quantitative hedge fund analyst.

        [DATA]
        Stock: {symbol}
        Timestamp: {datetime.now()}
        Current Price: {price}
        RSI: {df['RSI'].iloc[-1]:.2f}
        Volatility: {df['Volatility'].iloc[-1]:.2%}
        Trend (MA20): {trend}

        Recent News Headlines:
        {news_summary}

        Almanac Signals:
        - January Barometer: {jan_signal}
        - First 5 Trading Days: {five_signal}
        - Seasonality (Best 6 Months): {best6}
        - Presidential Cycle: {pres}

        [INSTRUCTIONS]
        1. Predict the price for next trading day (realistic, within ±10%)
        2. Provide:
        - target_price: realistic price (within ±10%)
        - confidence: 0 to 1
        3. Use:
        - technical indicators
        - news sentiment
        - Almanac Signals: (low weight)
        4. Be decisive

        [OUTPUT FORMAT - JSON ONLY]
        {{"target_price": 210.5,
        "confidence": 0.72,
        "reason": "max 15 sentences"
        }}
        """

        st.markdown('<div style="height: 150px;"></div>', unsafe_allow_html=True)
        btn_left, btn_center, btn_right = st.columns([1, 2, 1])

        with btn_center:
            run_llm_clicked = st.button("Run LLM Analysis", key="llm_button", use_container_width=True)

    # ---------- BUTTON ----------
    if run_llm_clicked:

        llm_text = run_llm(prompt)

        try:
            llm_data = json.loads(llm_text)

            llm_price = llm_data.get("target_price", price)
            llm_conf = llm_data.get("confidence", 0.5)
            llm_reason = llm_data.get("reason", "")

            llm_price = float(llm_price) if llm_price else price
            llm_conf = float(llm_conf) if llm_conf else 0.5

            llm_conf = min(max(llm_conf, 0), 1)

            if not llm_reason:
                llm_reason = "No reasoning provided"

            llm_reason = llm_reason[:2000]

        except Exception as e:
            st.error("LLM parsing failed")
            st.write(llm_text)

            llm_price = price
            llm_conf = 0.5
            llm_reason = "No analysis available"

        # ---------- ENSEMBLE ----------
        llm_conf = min(max(llm_conf, 0.2), 0.8)

        ensemble_price = (
            pred_price * (1 - llm_conf) +
            llm_price * llm_conf
        )
        st.session_state.ensemble_price = ensemble_price
        st.session_state.llm_price = llm_price
        st.session_state.pred_price = pred_price
        st.session_state.llm_reason = llm_reason
        st.session_state.llm_conf = llm_conf

    # ---------- UI ----------
    if "ensemble_price" in st.session_state:

        ensemble_price = st.session_state.ensemble_price
        llm_price = st.session_state.llm_price
        pred_price = st.session_state.pred_price
        llm_reason = st.session_state.llm_reason
        llm_conf = st.session_state.llm_conf

        # ---------- REASON ----------
        st.markdown("### 🧠 LLM Analysis")

        st.markdown(f"""
        <div class="themed-card" style="
            padding: 15px;
            border-radius: 10px;
            font-size: 15px;
            line-height: 1.6;
            margin-bottom:10px;
        ">
        {llm_reason}
        </div>
        """, unsafe_allow_html=True)

        # ---------- SIGNAL ----------
        signal_text = "BUY" if ensemble_price > price else "SELL"
        signal_class = "signal-buy" if signal_text == "BUY" else "signal-sell"
        arrow = "↑" if signal_text == "BUY" else "↓"

        st.markdown(f"""
        <div class="signal-card" style="
            padding: 25px;
            margin-bottom:10px;
        ">
            <div class="signal-card-title {signal_class}">{arrow} {signal_text}</div>
        </div>
        """, unsafe_allow_html=True)

        # ---------- VERTICAL CARDS ----------
        for title, value in [
            ("Ensemble Price", ensemble_price),
            ("LLM Price", llm_price),
            ("XGBoost Price", pred_price)
        ]:

            signal_text, signal_color = get_signal_style(value, price)

            st.markdown(f"""
            <div style="
                background: {card_bg};
                border: {card_border};
                padding: 20px;
                border-radius: 15px;
                margin-top:10px;
            ">
                <p style="color:{muted_text_color};">{title}</p>
                <h2>${value:.2f}</h2>
                <span style="color:{signal_color}; font-weight:600;">
                    {signal_text}
                </span>
            </div>
            """, unsafe_allow_html=True)
           
# ---------- HEATMAP ----------

with tab_heat:
    data = []

    for t in BIG_TECHS:

        try:

            d = yf.download(t, period="5d", progress=False)

            close = d["Close"]

            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            change = (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100

            data.append({"Ticker": t, "Change": float(change)})

        except:
            pass

    hdf = pd.DataFrame(data)

    fig = px.bar(
        hdf, x="Ticker", y="Change",
        color="Change", text="Change",
        color_continuous_scale="RdYlGn"
    )

    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(height=450, template=plotly_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font={'color': text_color})

    fig.update_xaxes(tickfont=dict(color=text_color))  
    fig.update_yaxes(tickfont=dict(color=text_color))  

    st.plotly_chart(fig, use_container_width=True)

# ---------- NEWS ----------

with tab_news:
    st.subheader(f"{symbol} News")

    for n in news[:10]:
        headline = n.get("headline", "No title")
        url = n.get("url", "#")
        summary = n.get("summary", "")
        date = datetime.fromtimestamp(n.get("datetime", 0)).strftime("%Y-%m-%d")

        st.markdown(f"**[{headline}]({url})**")
        st.write(summary)
        st.caption(date)
        st.divider()

# ---------- ALMANAC ----------

with tab_almanac:
    st.header("📅 Market Seasonality (Stock Trader's Almanac)")

    col1, col2, col3 = st.columns(3)

    st.metric("January Barometer", jan_signal)
    st.metric("First Five Days", five_signal)
    st.metric("Best Six Months", best6)

    st.subheader("Presidential Cycle")
    st.info(pres)
