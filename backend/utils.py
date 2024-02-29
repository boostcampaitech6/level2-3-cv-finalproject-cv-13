from PIL import Image
import numpy as np
import pandas as pd
import os,sys
import torch
from torchvision import transforms
from importlib import import_module

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from modeling.model import *
from modeling.dataloader import *
from model import *

def img_processing(imgs):
    img_array = []

    for img in imgs:
        image = img.convert("L")
        image_np = np.array(image)
        img_array.append(image_np)
    
    img_array = np.stack(img_array)
    img_array = np.stack((img_array,)*3, axis=1)
    img_array = torch.FloatTensor(img_array)
    img_array = img_array.unsqueeze(0)
    
    return img_array

#모델 불러오기
def load_model(saved_path, model_class, task, plane, device): 

    #동적으로 model.py를 import하고 원하는 class를 가져옴
    model_cls = getattr(import_module("model"), model_class) 
    model = model_cls()

    # 모델 가중치를 로드한다.
    model_path = os.path.join(saved_path, f"{task}_{plane}_best.pth")
    if os.path.exists(model_path): 
        model = torch.load(model_path, map_location=device)
    else:
        print("해당 경로에 모델 파일이 없습니다.")   

    return model

# fusion model
def fusion_model(saved_path, task, device):
    pass

# 개별 모델
def predict_task(input, path, model_class, task, plane):
    #gpu 확인
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # print(device)

    #모델 불러오기
    model = load_model(path, model_class, task, plane, device).to(device)
    model.eval()

    with torch.no_grad():
        input = input.to(device)
        predictions = model(input)
    probas = torch.sigmoid(predictions)
    proba = probas[0][1].item()
    return proba

def predict_percent(input):
    pass