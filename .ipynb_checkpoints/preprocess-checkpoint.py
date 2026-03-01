import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

img_tf = T.Compose([
    T.Resize((224,224)),
    T.ToTensor()
])

def process_image(img):
    if isinstance(img,str):
        img = Image.open(img).convert("RGB")
    return img_tf(img).unsqueeze(0)


def process_prices(prices):
    arr = np.array(prices).astype(np.float32)
    return torch.tensor(arr).view(1,-1,1)