from PIL import Image
import numpy as np
from model import *

import pandas as pd
import os
import torch
from torchvision import transforms
from importlib import import_module

def img_processing(imgs):
    img_array = []

    for img in imgs:
        image = img.convert("L")
        image_np = np.array(image)
        img_array.append(image_np)
    
    return np.stack(img_array)

#모델 불러오기
def load_model(saved_path, model_class, device): 
    # model_class = "EfficientNet_b2"

    #동적으로 model.py를 import하고 원하는 class를 가져옴
    model_cls = getattr(import_module("model"), model_class) 
    model = model_cls(18) #class 개수

    # 모델 가중치를 로드한다.
    model_path = os.path.join(saved_path, model_class+"_best.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

def predict_image(input):
    #gpu 확인
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    #모델 불러오기
    model = load_model("./model", "EfficientNet_b2", device).to(device)
    model.eval()

    ## 임시 모델에 전달하기 위해 인풋 변경
    image = torch.tensor(input)
    image = image.unsqueeze(0)
    # print(image)
    # print(image.shape)
    image = image.float() / 255.0
    ##

    with torch.no_grad():
        predictions = model(image)
    # print(predictions)
    prediction = torch.argmax(predictions).item()
    # print(prediction)
    return prediction

