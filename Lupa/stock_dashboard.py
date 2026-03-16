
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

from sklearn.ensemble import RandomForestRegressor


# ---------- CONFIG ----------

FINNHUB_API_KEY = st.secrets["FINNHUB_API_KEY"]
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

BIG_TECHS = [
"AAPL","MSFT","NVDA","AMZN",
"META","TSLA","GOOGL","AMD"
]

st.set_page_config(
page_title="Lupa AI Stock Terminal",
layout="wide",
page_icon="📈"
)

# ---------- STYLE ----------

st.markdown("""
<style>
.stApp{
background:#0f172a;
color:white;
}
section[data-testid="stSidebar"]{
background:#111827;
}
[data-testid="stMetricValue"]{
font-size:28px;
}
</style>
""", unsafe_allow_html=True)

# ---------- LOGO ----------

logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
logo = Image.open(logo_path)

col_logo,col_title = st.columns([1,4])

with col_logo:
    st.image(logo,width=120)

with col_title:
    st.title("Lupa AI Stock Terminal")


# ---------- SIDEBAR STATE ----------

if "ticker" not in st.session_state:
    st.session_state.ticker="AAPL"

if "bigtech" not in st.session_state:
    st.session_state.bigtech="AAPL"


def ticker_changed():
    ticker = st.session_state.ticker.upper()
    if ticker in BIG_TECHS:
        st.session_state.bigtech = ticker


def bigtech_changed():
    st.session_state.ticker = st.session_state.bigtech


# ticker input
st.sidebar.text_input(
"Ticker",
key="ticker",
on_change=ticker_changed
)

# big tech selector
st.sidebar.radio(
"Big Tech",
BIG_TECHS,
key="bigtech",
on_change=bigtech_changed
)

symbol = st.session_state.ticker.upper()

period = st.sidebar.selectbox(
"Period",
["3mo","6mo","1y","2y","5y"],
index=2
)


# ---------- DATA ----------

@st.cache_data
def load_data(symbol,period):

    stock = yf.Ticker(symbol)
    df = stock.history(period=period)

    df["MA20"] = df["Close"].rolling(20).mean()

    delta = df["Close"].diff()

    gain=(delta.where(delta>0,0)).rolling(14).mean()
    loss=(-delta.where(delta<0,0)).rolling(14).mean()

    df["RSI"] = 100-(100/(1+gain/loss))

    df["Returns"] = df["Close"].pct_change()

    df["Volatility"] = df["Returns"].rolling(20).std()*np.sqrt(252)

    return df


df = load_data(symbol,period)

if df.empty:
    st.error("Ticker not found")
    st.stop()

price = df["Close"].iloc[-1]
ret = df["Returns"].iloc[-1]


# ---------- HEADER ----------

st.markdown(f"## 📊 {symbol} Market Overview")

col1,col2,col3,col4 = st.columns(4)

with col1:
    st.metric("Price",f"${price:.2f}",f"{ret:.2%}")

with col2:
    trend="Bullish" if price>df["MA20"].iloc[-1] else "Bearish"
    st.metric("Trend",trend)

with col3:
    st.metric("Volatility",f"{df['Volatility'].iloc[-1]:.2%}")

with col4:
    st.metric("RSI",f"{df['RSI'].iloc[-1]:.1f}")


# ---------- MARKET SENTIMENT ----------

sentiment = 50 + ret*100

fig_sent = go.Figure(go.Indicator(
mode="gauge+number",
value=sentiment,
title={'text':"Market Sentiment"},
gauge={
'axis':{'range':[0,100]},
'bar':{'color':"#3b82f6"},
'steps':[
{'range':[0,40],'color':"#ef4444"},
{'range':[40,60],'color':"#facc15"},
{'range':[60,100],'color':"#22c55e"}
]
}
))

st.plotly_chart(fig_sent,use_container_width=True)


# ---------- CHART ----------

def create_chart(df):

    fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.75,0.25]
    )

    fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    increasing_line_color="#22c55e",
    decreasing_line_color="#ef4444"
    ),row=1,col=1)

    fig.add_trace(go.Scatter(
    x=df.index,
    y=df["MA20"],
    line=dict(color="#60a5fa",width=2),
    name="MA20"
    ),row=1,col=1)

    fig.add_trace(go.Bar(
    x=df.index,
    y=df["Volume"],
    marker_color="rgba(120,160,255,0.3)"
    ),row=2,col=1)

    fig.update_layout(
    template="plotly_dark",
    height=650,
    hovermode="x unified"
    )

    return fig


# ---------- AI MODEL (RandomForest instead of LSTM) ----------

def lstm_forecast(df, window=10):

    prices = df["Close"].values

    X = []
    y = []

    for i in range(window, len(prices)):
        X.append(prices[i-window:i])
        y.append(prices[i])

    X = np.array(X)
    y = np.array(y)

    model = RandomForestRegressor(n_estimators=100)

    model.fit(X, y)

    last_window = prices[-window:].reshape(1, -1)

    pred = model.predict(last_window)

    return float(pred[0])


# ---------- TABS ----------

tab_chart,tab_ai,tab_heat,tab_news = st.tabs([
"📊 Chart",
"🤖 AI Forecast",
"🌎 Heatmap",
"📰 News"
])


# ---------- CHART TAB ----------

with tab_chart:

    fig=create_chart(df)
    st.plotly_chart(fig,use_container_width=True)


# ---------- AI TAB ----------

with tab_ai:

    col1,col2 = st.columns(2)

    with col1:

        st.subheader("AI Price Prediction")

        lstm_price = lstm_forecast(df)

        direction = "Bullish" if lstm_price > price else "Bearish"

        st.metric(
        "Predicted Price",
        f"${lstm_price:.2f}",
        direction
        )

    with col2:

        st.subheader("LLM Analysis")

        trend="Bullish" if price>df["MA20"].iloc[-1] else "Bearish"

        prompt=f"""
You are a professional quantitative analyst.

Stock: {symbol}

Price: {price}
RSI: {df['RSI'].iloc[-1]:.2f}
Volatility: {df['Volatility'].iloc[-1]:.2%}
Trend: {trend}

Give a short outlook.
"""

        if st.button("Run LLM Analysis"):

            response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role":"user","content":prompt}]
            )

            st.write(response.choices[0].message.content)

    st.divider()

    st.subheader("AI Comparison")

    st.write(f"""
Current Price: **${price:.2f}**

AI Prediction: **${lstm_price:.2f}**

Expected Move: **{lstm_price-price:.2f}**
""")


# ---------- HEATMAP ----------

with tab_heat:

    data=[]

    for t in BIG_TECHS:

        try:

            d=yf.download(t,period="5d",progress=False)

            close=d["Close"]

            if isinstance(close,pd.DataFrame):
                close=close.iloc[:,0]

            change=(close.iloc[-1]-close.iloc[0])/close.iloc[0]*100

            data.append({
            "Ticker":t,
            "Change":float(change)
            })

        except:
            pass

    hdf=pd.DataFrame(data)

    fig=px.bar(
    hdf,
    x="Ticker",
    y="Change",
    color="Change",
    text="Change",
    color_continuous_scale="RdYlGn"
    )

    fig.update_traces(texttemplate="%{text:.2f}%",textposition="outside")

    fig.update_layout(template="plotly_dark",height=450)

    st.plotly_chart(fig,use_container_width=True)


# ---------- NEWS ----------

with tab_news:

    st.subheader(f"{symbol} News")

    today=datetime.today().strftime("%Y-%m-%d")
    last_week=(datetime.today()-timedelta(days=7)).strftime("%Y-%m-%d")

    news=finnhub_client.company_news(symbol,_from=last_week,to=today)

    for n in news[:10]:

        st.markdown(f"**[{n['headline']}]({n['url']})**")

        st.write(n.get("summary",""))

        st.caption(
        datetime.fromtimestamp(n["datetime"]).strftime("%Y-%m-%d")
        )

        st.divider()
