import os, json, random
import numpy as np
import yfinance as yf
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

from utils import *

# reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# model
def build_model(lstm_shape,cnn_shape):

    li=Input(shape=lstm_shape)
    l=LSTM(64)(li)

    ci=Input(shape=cnn_shape)
    c=Conv2D(32,(3,3),activation="relu")(ci)
    c=MaxPooling2D()(c)
    c=Flatten()(c)

    x=concatenate([l,c])
    x=Dense(64,activation="relu")(x)
    out=Dense(1,activation="sigmoid")(x)

    m=Model([li,ci],out)
    m.compile(Adam(0.001),"binary_crossentropy",["accuracy"])
    return m

# training pipeline
def train(ticker):

    set_seed()

    df=yf.download(ticker,start="2015-01-01",auto_adjust=False)
    df.columns=df.columns.get_level_values(0)

    df=build_features(df)

    generate_charts(df)

    Xl,Xc,y,scaler,split=build_dataset(df)

    model=build_model(
        (Xl.shape[1],Xl.shape[2]),
        (128,128,3)
    )

    model.fit(
        [Xl[:split],Xc[:split]],
        y[:split],
        epochs=10,
        batch_size=8,
        verbose=1
    )

    preds=(model.predict([Xl[split:],Xc[split:]])>0.5).astype(int)
    acc=accuracy_score(y[split:],preds)

    os.makedirs("models",exist_ok=True)
    model.save(f"models/{ticker}.keras")

    print("Saved model:",ticker,"Accuracy:",acc)

# run
train("AAPL")