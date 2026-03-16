
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def lstm_forecast(df, window=60):

    prices = df["Close"].values.reshape(-1,1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    X=[]
    y=[]

    for i in range(window,len(scaled)):
        X.append(scaled[i-window:i])
        y.append(scaled[i])

    X=np.array(X)
    y=np.array(y)

    model=Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(window,1)))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(optimizer="adam",loss="mse")

    model.fit(X,y,epochs=3,batch_size=32,verbose=0)

    last_window=scaled[-window:]
    last_window=last_window.reshape(1,window,1)

    pred_scaled=model.predict(last_window,verbose=0)
    pred=scaler.inverse_transform(pred_scaled)

    return float(pred[0][0])
