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
# [新增] 在侧边栏添加白天/黑夜切换开关
dark_mode = st.sidebar.toggle("Night Mode", value=True)

# [新增] 根据模式定义颜色变量，白天模式强制使用纯黑 (#000000) 确保高对比度
if dark_mode:
    bg_style = "radial-gradient(circle at 50% 30%, rgba(255,255,255,0.05), transparent 60%), radial-gradient(circle at center, #1e293b 0%, #020617 100%)"
    sidebar_bg = "#020617"
    text_color = "white"
    metric_bg = "rgba(255,255,255,0.05)"
    plotly_template = "plotly_dark"
    grid_color = "rgba(255,255,255,0.1)"
else:
    bg_style = "#f0f2f6"
    sidebar_bg = "#ffffff"
    text_color = "#000000"  # [修改] 白天模式强制纯黑
    metric_bg = "#ffffff"
    plotly_template = "plotly_white"
    grid_color = "rgba(0,0,0,0.1)"

# ---------- STYLE ----------

# [修改] 使用 f-string 动态注入颜色变量，并增加对按钮、指标数字、输入框的强制样式覆盖
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

/* [新增] 强制所有层级的文字、标签颜色，确保在白天模式下可见 */
h1, h2, h3, h4, h5, p, label, span, div {{
    color: {text_color} !important;
}}

/* [新增] 专项修复：指标数字 (Metric Value) 颜色 */
[data-testid="stMetricValue"] div {{
    color: {text_color} !important;
}}

/* [新增] 专项修复：按钮内部文字颜色保持白色 (以适配深色按钮背景) */
.stButton > button p {{
    color: white !important;
    font-weight: 700 !important;
}}

/* [新增] 专项修复：Tab 标签页文字颜色 */
button[data-baseweb="tab"] div {{
    color: {text_color} !important;
}}

/* [新增] 专项修复：侧边栏输入框和下拉框文字颜色保持白色 */
.stTextInput input, .stSelectbox div[data-baseweb="select"] {{
    color: white !important;
    -webkit-text-fill-color: white !important;
}}

</style>
""", unsafe_allow_html=True)
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


def bigtech_changed():
    st.session_state.ticker = st.session_state.bigtech


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
    title={'text': "Market Sentiment", 'font': {'color': text_color}},  # [修改] 显式设置标题颜色
    gauge={
        'axis': {'range': [0, 100], 'tickcolor': text_color, 'tickfont': {'color': text_color}},  # [新增] 设置刻度文字颜色
        'bar': {'color': "#3b82f6"},
        'steps': [
            {'range': [0, 40], 'color': "#ef4444"},
            {'range': [40, 60], 'color': "#facc15"},
            {'range': [60, 100], 'color': "#22c55e"}
        ]
    }
))

# [修改] 更新仪表盘，确保内部数字和字体颜色随主题变化
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

    # [修改] 更新图表轴标及网格颜色，确保在白天模式下字符可见
    fig.update_layout(
        template=plotly_template,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': text_color}
    )

    fig.update_xaxes(tickfont=dict(color=text_color), gridcolor=grid_color)  # [新增] 强制 X 轴刻度变色
    fig.update_yaxes(tickfont=dict(color=text_color), gridcolor=grid_color)  # [新增] 强制 Y 轴刻度变色

    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date"
        )
    )

    return fig


# ---------- SEASONALITY ----------

def best_six_months():
    month = datetime.now().month

    if month in [11, 12, 1, 2, 3, 4]:
        return "Bullish Season"
    else:
        return "Weak Season"


# ---------- AI MODEL ----------

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
        messages=[{"role": "user", "content": prompt}]
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

# ---------- AI ----------

with tab_ai:
    col1, col2 = st.columns(2)

    with col1:

        st.subheader("XGBoost Prediction")

        pred_price = price_forecast(df)

        direction = "Bullish" if pred_price > price else "Bearish"

        st.metric("Predicted Price", f"${pred_price:.2f}", direction)

    with col2:

        st.subheader("LLM Analysis")

        prompt = f"""
You are a professional quantitative analyst.

Stock: {symbol}
Price: {price}
RSI: {df['RSI'].iloc[-1]:.2f}
Volatility: {df['Volatility'].iloc[-1]:.2%}
Trend: {trend}

Give short outlook.
"""

        if st.button("Run LLM Analysis", key="llm_button"):

            llm_text = run_llm(prompt)

            st.write(llm_text)

            llm_signal = "bullish" if "bullish" in llm_text.lower() else "bearish"
            model_signal = "bullish" if pred_price > price else "bearish"
            trend_signal = "bullish" if trend == "Bullish" else "bearish"

            # NEW
            best6 = best_six_months()
            season_signal = "bullish" if best6 == "Bullish Season" else "neutral"

            votes = [llm_signal, model_signal, trend_signal]

            if season_signal != "neutral":
                votes.append(season_signal)

            bullish = votes.count("bullish")
            bearish = votes.count("bearish")

            if bullish > bearish:
                final_signal = "BUY"
            elif bearish > bullish:
                final_signal = "SELL"
            else:
                final_signal = "HOLD"

            confidence = max(bullish, bearish) / len(votes)

            st.subheader("AI Trading Signal")

            st.write(f"""
            Trend: **{trend_signal.upper()}**

            XGBoost: **{model_signal.upper()}**

            LLM: **{llm_signal.upper()}**

            Seasonality: **{best6}**

            Signal: **{final_signal}**

            Confidence: **{confidence:.0%}**
            """)

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
    # [修改] 更新轴标及字体，确保热力图在白天模式下可见
    fig.update_layout(height=450, template=plotly_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font={'color': text_color})

    fig.update_xaxes(tickfont=dict(color=text_color))  # [新增] 强制热力图 X 轴变色
    fig.update_yaxes(tickfont=dict(color=text_color))  # [新增] 强制热力图 Y 轴变色

    st.plotly_chart(fig, use_container_width=True)

# ---------- NEWS ----------

with tab_news:
    st.subheader(f"{symbol} News")

    today = datetime.today().strftime("%Y-%m-%d")
    last_week = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")

    news = finnhub_client.company_news(symbol, _from=last_week, to=today)

    for n in news[:10]:
        st.markdown(f"**[{n['headline']}]({n['url']})**")
        st.write(n.get("summary", ""))

        st.caption(
            datetime.fromtimestamp(n["datetime"]).strftime("%Y-%m-%d")
        )

        st.divider()

# ---------- ALMANAC ----------

with tab_almanac:
    st.header("📅 Market Seasonality (Stock Trader's Almanac)")

    col1, col2, col3 = st.columns(3)

    # ---------- January Barometer ----------

    spy = yf.download("SPY", period="2y", progress=False)

    jan = spy[spy.index.month == 1]

    if len(jan) > 5:

        close = jan["Close"]

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        jan_return = float((close.iloc[-1] / close.iloc[0]) - 1)

        jan_signal = "Bullish" if jan_return > 0 else "Bearish"

    else:

        jan_signal = "Waiting for January"

    # ---------- First Five Days ----------

    jan5 = jan.head(5)

    if len(jan5) == 5:

        close = jan5["Close"]

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        jan5_return = float((close.iloc[-1] / close.iloc[0]) - 1)

        five_signal = "Bullish" if jan5_return > 0 else "Bearish"

    else:

        five_signal = "Not available"

    # ---------- Best Six Months ----------

    month = datetime.now().month

    if month in [11, 12, 1, 2, 3, 4]:

        best6 = "Bullish Season"

    else:

        best6 = "Weak Season"

    with col1:

        st.metric(
            "January Barometer",
            jan_signal
        )

    with col2:

        st.metric(
            "First Five Days",
            five_signal
        )

    with col3:

        st.metric(
            "Best Six Months",
            best6
        )

    st.divider()

    # ---------- Presidential Cycle ----------

    year = datetime.now().year

    cycle = (year - 2024) % 4

    if cycle == 0:
        pres = "Election Year"

    elif cycle == 1:
        pres = "Post Election"

    elif cycle == 2:
        pres = "Midterm Weakness"

    else:
        pres = "Pre Election Bullish"

    st.subheader("Presidential Cycle")

    st.info(pres)