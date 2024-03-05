from PIL import Image
import numpy as np
import pandas as pd
import os
import torch
from importlib import import_module
import pickle
from sklearn.linear_model import LogisticRegression

from model import MRNet
from config import config


from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def data_processing(path):
    
    img_array = np.load(path)

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
def fusion_model(saved_path, task):
    model_path = os.path.join(saved_path, f"lr_{task}.pkl")
    if os.path.exists(model_path): 
        with open(model_path, 'rb') as f: 
            model = pickle.load(f)
    else:
        print("해당 경로에 모델 파일이 없습니다.")   

    return model

# 개별 모델
def predict_disease(input, path, model_class, task, plane):
    #gpu 확인
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    #모델 불러오기
    model = load_model(path, model_class, task, plane, device).to(device)
    model.eval()

    with torch.no_grad():
        input = input.to(device)
        predictions = model(input)
    
    probas = torch.sigmoid(predictions)
    proba = probas[0][1].item()
    return proba

def predict_percent(input, path, task):
    # feature_names = ['axial','coronal','sagittal']
    input = np.array(input).reshape(1,-1)
    lr_model = fusion_model(path, task)
    proba = lr_model.predict_proba(input)[:, 1]

    return proba

def grad_cam_inference(input, path, model_class, task, plane):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    grad_dir = os.path.join('gradcam', task)

    #모델 불러오기
    model = load_model(path, model_class, task, plane, device).to(device)
    model.eval()

    # target_layers = model.target
    target_layers = [model.pretrained_model.features[-1]]
    score_li = []

    with GradCAM(model=model, target_layers=target_layers) as cam:
        targets = [BinaryClassifierOutputTarget(1)]
        
        cam_result_list = cam(
                            input_tensor=input.float(), targets=targets, 
                            aug_smooth=True, eigen_smooth=True
                                )
        
        #camscore처리
        for array in cam_result_list:
            score_li.append(array.sum())
        max_idx = score_li.index(max(score_li))

        print(f'Generating Grad-CAM Images about {task}-{plane}')
        original_image = torch.squeeze(input, dim=0).permute(0, 2, 3, 1).cpu().numpy()[max_idx] / 255.0
        cam_result = cam_result_list[max_idx]
        visualization = show_cam_on_image(original_image, cam_result) 
        visualization_image = Image.fromarray(visualization)

        if not os.path.exists(grad_dir):
            os.makedirs(grad_dir)
        
        grad_path = os.path.join(grad_dir, plane +'.png')
        if os.path.exists(grad_path):
            os.remove(grad_path)
        
        visualization_image.save(grad_path)

    return max_idx, score_li