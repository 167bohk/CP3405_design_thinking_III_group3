
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def llm_prediction(symbol, price, rsi, volatility, trend):

    prompt=f"""
You are a professional quantitative analyst.

Stock: {symbol}
Price: {price}
RSI: {rsi}
Volatility: {volatility}
Trend: {trend}

Predict short term outlook (1-5 days).
Return direction, confidence and explanation.
"""

    response=client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role":"user","content":prompt}]
    )

    return response.choices[0].message.content
