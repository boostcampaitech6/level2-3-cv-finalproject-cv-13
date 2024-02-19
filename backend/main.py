from fastapi import FastAPI, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
import os
import numpy as np
import base64

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

@app.post("/input")
async def receiveFile(file: list[UploadFile]):
    IMAGE_ROOT = 'input_img'
    img_json = {}

    #1.이미지 저장
    img_list = []
    for f in file:
        print(f.filename)
        image = Image.open(f.file)
        if not os.path.exists(IMAGE_ROOT):
            os.makedirs(IMAGE_ROOT)
        image.save(os.path.join(IMAGE_ROOT, f.filename), 'PNG')
        img_list.append(image)
    
    # plane = ['axial', 'coronal', 'sagittal'] -> 형식 바꿔야함
    img_json['img'] = img_list

    #2. 전처리
    img_json['numpy'] = img_processing(img_json['img'])
    print(img_json['numpy'].shape)

    #3. 모델 추론
    img_json['result'] = predict_image(img_json['numpy'])
    print(img_json['result'])
    
    """
    4. img_json['grad_cam']에 gradcam 결과값 입력
    or result_img에 gradcam 이미지 저장...
    """
    return img_json

@app.get("/outputoriginal")
async def outputFile():
    IMAGE_ROOT = 'input_img' 
    output_bytes = []
    image_paths = os.listdir(IMAGE_ROOT)
    for path in image_paths:
        
        # arr = np.load(os.path.join(IMAGE_ROOT, path))
        # image = Image.fromarray(arr[0, :, :].astype(np.uint8))
        # image.save(os.path.join(IMAGE_ROOT, path), 'PNG')

        with open(os.path.join(IMAGE_ROOT, path), 'rb') as img:
            base64_string = base64.b64encode(img.read())

        headers = {'Content-Disposition': 'inline; filename="test.png"'}
        output_bytes.append(Response(base64_string, headers=headers, media_type='image/png'))
    return output_bytes


@app.get("/outputgradcam")
async def outputFile():
    IMAGE_ROOT = 'sampleimages2'
    output_bytes = []
    image_paths = os.listdir(IMAGE_ROOT)
    for path in image_paths:
        
        # arr = np.load(os.path.join(IMAGE_ROOT, path))
        # image = Image.fromarray(arr[0, :, :].astype(np.uint8))
        # image.save(os.path.join(IMAGE_ROOT, path), 'PNG')

        with open(os.path.join(IMAGE_ROOT, path), 'rb') as img:
            base64_string = base64.b64encode(img.read())

        headers = {'Content-Disposition': 'inline; filename="test.jpg"'}
        output_bytes.append(Response(base64_string, headers=headers, media_type='image/jpg'))
    return output_bytes