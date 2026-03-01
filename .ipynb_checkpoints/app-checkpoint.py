import gradio as gr
import torch
from fusion_model import CNNModel,LSTMModel,FusionModel
from preprocess import process_image,process_prices

# ===== load models =====
cnn = CNNModel()
lstm = LSTMModel()

cnn.load_state_dict(torch.load("weights/cnn.pth",map_location="cpu"))
lstm.load_state_dict(torch.load("weights/lstm.pth",map_location="cpu"))

model = FusionModel(cnn,lstm)
model.load_state_dict(torch.load("weights/fusion_head.pth",map_location="cpu"))
model.eval()

# ===== predict =====
def predict(img,price_str):

    prices = [float(x) for x in price_str.split(",")]

    img_t = process_image(img)
    seq_t = process_prices(prices)

    with torch.no_grad():
        prob = model(img_t,seq_t).item()

    signal = "BUY 📈" if prob>0.5 else "SELL 📉"

    return {"signal":signal,"probability":round(prob,4)}

# ===== UI =====
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil"),
        gr.Textbox(label="Last 30 prices (comma separated)")
    ],
    outputs="json",
    title="AI Trading Signal System"
)

demo.launch()