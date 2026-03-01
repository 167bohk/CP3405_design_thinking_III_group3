import yfinance as yf
import numpy as np
from tensorflow.keras.models import load_model
from utils import *

def predict(ticker):

    model=load_model(f"models/{ticker}.keras")

    df=yf.download(ticker,period="6mo",auto_adjust=False)
    df.columns=df.columns.get_level_values(0)

    df=build_features(df)
    generate_charts(df)

    Xl,Xc,y,scaler,_=build_dataset(df)

    prob=model.predict([Xl[-1:],Xc[-1:]])[0][0]

    return {
        "prediction":"UP" if prob>0.5 else "DOWN",
        "confidence":float(prob)
    }

print(predict("AAPL"))