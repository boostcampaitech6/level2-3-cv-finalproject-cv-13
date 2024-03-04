from fastapi import FastAPI, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from PIL import Image
import os
import numpy as np
import base64
import json
import shutil

from utils import *

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

@app.post("/input/dicom/{plane}")
async def receiveDICOM(plane: str, file: UploadFile):
    pass

@app.post("/input/{plane}") # ex) /input/axial
async def receiveFile(plane: str, file: list[UploadFile]):
    IMAGE_ROOT = os.path.join('original', plane)
    if os.path.exists(IMAGE_ROOT): 
        shutil.rmtree(IMAGE_ROOT) # 이미 폴더 있으면 삭제
    os.makedirs(IMAGE_ROOT)

    # 이미지 서버에 저장
    for f in file:
        print(f.filename)
        image = Image.open(f.file)
        image.save(os.path.join(IMAGE_ROOT, f.filename), 'PNG')

    return {plane : "success"}

@app.get("/inference")
async def inference():
    planes = ['axial', 'coronal', 'sagittal']
    tasks = ['abnormal', 'acl', 'meniscus']
    input_dict = {}

    # JSON template 불러오기
    with open('./template.json', 'r') as file:
        result_dict = json.load(file)

    for task in tasks:
        result_dict['percent']['labels'].append(task)
        res = []

        for plane in planes:
            IMAGE_ROOT = os.path.join('original', plane)
            image_paths = os.listdir(IMAGE_ROOT)
            img_list = []

            # 1. 이미지 불러오기
            for path in image_paths:
                image = Image.open(os.path.join(IMAGE_ROOT, path))
                img_list.append(image)

            # 2. 전처리
            input_tensor = img_processing(img_list)
            grad_cam_inference(input_tensor, "./models", "MRNet", task, plane)
            
            # 3. 개별 모델 추론
            res.append(predict_task(input_tensor, "./models", "MRNet", task, plane))

        # 4. fusion 모델 추론
        proba = {}
        proba['y'] = task
        fusion_res = predict_percent(res,"./models", task)
        proba['x'] = round((fusion_res[0] * 100),1)
        result_dict['percent']['datasets'].append(proba)

        # 5. img_json['grad_cam']에 gradcam 결과값 입력
        # or result_img에 gradcam 이미지 저장...
        
    
    #6. 추론 결과 json으로 저장
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