from fastapi import FastAPI, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from PIL import Image
import os
import numpy as np
import base64
import json

from utils import *
from model import *

origins = [
    "http://localhost:3000",
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello"}

@app.post("/input/{plane}") # ex) /input/axial
async def receiveFile(plane: str, file: list[UploadFile]):
    IMAGE_ROOT = os.path.join('original', plane)
    # img_json = {}
    # img_list = []

    #1.이미지 저장
    for f in file:
        print(f.filename)
        image = Image.open(f.file)
        #이미 존재하는 폴더도 삭제하고 새로 만들어야 할 수도?
        if not os.path.exists(IMAGE_ROOT): 
            os.makedirs(IMAGE_ROOT)
        image.save(os.path.join(IMAGE_ROOT, f.filename), 'PNG')
        # img_list.append(image)
    
    return {plane : "success"}

@app.get("/inference")
async def inference():
    plane = ['axial', 'coronal', 'sagittal']
    input_dict = {}
    # JSON 파일 불러오기
    with open('./template.json', 'r') as file:
        result_dict = json.load(file)

    for p in plane:
        IMAGE_ROOT = os.path.join('original', p)
        image_paths = os.listdir(IMAGE_ROOT)
        img_list = []

        #1. 이미지 불러오기
        for path in image_paths:
            image = Image.open(os.path.join(IMAGE_ROOT, path))
            img_list.append(image)

            #2. 전처리
            input_dict[p] = img_processing(img_list)
        
        print(f"{p}: {input_dict[p].shape}")

        """
        #3. 모델 추론
        result_dict['result'] = predict_image(img_json['numpy'])
        print(img_json['result'])
        print(img_json)

        4. img_json['grad_cam']에 gradcam 결과값 입력
        or result_img에 gradcam 이미지 저장...
        """
    
    #4. 추론 결과 json으로 저장
    with open('./result.json','w') as f:
        json.dump(result_dict, f, indent=4)

    return {"inference" : "success"}


@app.get("/totalresult")
async def outputJSON():
    # JSON 파일 불러오기
    with open('./result.json', 'r') as file:
        result_dict = json.load(file)
    
    return result_dict['percent']


@app.get("/output/{task}/{plane}/{method}") # ex) ouput/abnormal/axial/orignial
async def outputFile(task: str, plane:str, method:str ):
    if method == "original": #original or gradcam
        IMAGE_ROOT = os.path.join(method,plane) 
    elif method == "gradcam":
        IMAGE_ROOT = os.path.join(method, task, plane)
    
    #이미지 정보
    output_bytes = []
    image_paths = os.listdir(IMAGE_ROOT)
    for path in image_paths:

        with open(os.path.join(IMAGE_ROOT, path), 'rb') as img:
            base64_string = base64.b64encode(img.read())

        headers = {'Content-Disposition': 'inline; filename="test.png"'}
        output_bytes.append(Response(base64_string, headers=headers, media_type='image/png'))
    
    #task-plane importance 정보
    with open('./result.json', 'r') as file:
        result_dict = json.load(file)
    info = result_dict[task][plane]

    result_info = {}
    #info + 이미지
    result_info['info'] = info
    result_info['img'] = output_bytes

    return result_info

@app.get("/result/{task}/{method}")
async def resultFile(task:str, method:str):
    if method == "original": #original or gradcam
        IMAGE_ROOT = os.path.join(task, method) 
    elif method == "gradcam":
        IMAGE_ROOT = os.path.join(task, method)
    
    output_bytes = []
    image_paths = os.listdir(IMAGE_ROOT)
    for path in image_paths:

        with open(os.path.join(IMAGE_ROOT, path), 'rb') as img:
            base64_string = base64.b64encode(img.read())

        headers = {'Content-Disposition': 'inline; filename="test.png"'}
        output_bytes.append(Response(base64_string, headers=headers, media_type='image/png'))

    return output_bytes