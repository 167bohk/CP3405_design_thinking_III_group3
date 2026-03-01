import os, shutil, numpy as np, pandas as pd, mplfinance as mpf
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

# ---------- feature engineering ----------
def build_features(df):

    df["ret"]=df["Close"].pct_change()
    df["vol"]=df["ret"].rolling(10).std()

    df["month"]=df.index.month
    df["dow"]=df.index.dayofweek
    df["cycle"]=(df.index.year%4==2).astype(int)

    df["sin"]=np.sin(2*np.pi*df["month"]/12)
    df["cos"]=np.cos(2*np.pi*df["month"]/12)

    df["target"]=(df["Close"].shift(-1)>df["Close"]).astype(int)

    return df.dropna()

# ---------- image loader ----------
def load_img(path,size=128):

    img=Image.open(path)
    if img.mode!="RGB":
        img=img.convert("RGB")

    img=img.resize((size,size))
    return np.array(img)/255.0

# ---------- generate charts ----------
def generate_charts(df,window=60,folder="charts"):

    shutil.rmtree(folder,ignore_errors=True)
    os.makedirs(folder)

    for i in range(window,len(df)):
        mpf.plot(
            df.iloc[i-window:i],
            type="candle",
            style="charles",
            volume=False,
            savefig=f"{folder}/{i}.png"
        )

# ---------- dataset ----------
def build_dataset(df,split=0.8,window=60):

    feats=["Close","Volume","ret","vol","month","dow","cycle","sin","cos"]

    split_i=int(len(df)*split)

    scaler=MinMaxScaler()
    scaler.fit(df[feats].iloc[:split_i])
    scaled=scaler.transform(df[feats])

    Xl,Xc,y=[],[],[]

    for i in range(window,len(df)):
        Xl.append(scaled[i-window:i])
        Xc.append(load_img(f"charts/{i}.png"))
        y.append(df["target"].iloc[i])

    return np.array(Xl),np.array(Xc),np.array(y),scaler,split_i