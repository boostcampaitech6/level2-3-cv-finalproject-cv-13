from fastapi import FastAPI, UploadFile, Response, HTTPException
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
from schemas import DICOMRequest, resultResponse, DiseaseResult
from dcm_convert import convert_dcm_to_numpy
from config import config

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
async def receiveFile(plane:str, file: list[UploadFile]):
    UPLOAD_FOLDER = os.path.join(config.orign_path, plane)

    if os.path.exists(UPLOAD_FOLDER): 
        shutil.rmtree(UPLOAD_FOLDER) # 이미 폴더 있으면 삭제
    os.makedirs(UPLOAD_FOLDER)

    # DICOM 서버에 저장
    for f in file:
        ext = os.path.splitext(f.filename)[1]
        if ext.lower() not in config.ext:
            raise HTTPException(status_code=400, detail="지원되지 않는 파일 형식입니다.")

        try:
            file_path = os.path.join(UPLOAD_FOLDER, f.filename)
            with open(file_path, "wb") as file_object:
                file_object.write(f.file.read())

                _, npy_array = convert_dcm_to_numpy(file_path)
                np.save(file_path.replace(ext,'.npy'), npy_array)
                
                #test
                check = np.load(file_path.replace(ext,'.npy'))
                print(check.shape)
            return JSONResponse(status_code=200, content={ plane: "success"})
        except Exception as e:
            raise HTTPException(status_code=500, content={ plane: "파일 업로드에 실패했습니다.", "error": str(e)})
        

@app.get("/inference")
async def inference():
    planes = config.planes
    diseases = config.diseases

    # JSON template 불러오기
    with open('./template.json', 'r') as file:
        result_dict = json.load(file)

    for disease in diseases:
        result_dict['percent']['labels'].append(disease)
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
            
            # 3. 개별 모델 추론
            res.append(predict_disease(input_tensor, "./models", "MRNet", disease, plane))
            grad_cam_inference(input_tensor, "./models", "MRNet", disease, plane)

            # 5. img_json['grad_cam']에 gradcam 결과값 입력
            # or result_img에 gradcam 이미지 저장...

        # 4. fusion 모델 추론
        proba = {}
        proba['y'] = disease
        fusion_res = predict_percent(res,"./models", disease)
        proba['x'] = round((fusion_res[0] * 100),1)
        result_dict['percent']['datasets'].append(proba)

    #6. 추론 결과 json으로 저장
    with open('./result.json','w') as f:
        json.dump(result_dict, f, indent=4)

    return {"inference" : "success"}

# 전체 결과
@app.get("/result")
async def outputJSON() -> resultResponse:
    # JSON 파일 불러오기
    with open('./result.json', 'r') as file:
        result_dict = json.load(file)
    result = result_dict['percent']
    return resultResponse(labels=result["labels"], datasets=result["datasets"])

# 원본이미지 + 질병에 따른 사진 별 score 그래프
@app.get("/output/{disease}/{plane}") # ex) ouput/abnormal/axial
async def outputFile(disease: str, plane:str, method:str) -> DiseaseResult:
    IMAGE_ROOT = os.path.join("original", plane)
    #이미지 정보
    output_bytes = []
    image_paths = os.listdir(IMAGE_ROOT)
    for path in image_paths:

        with open(os.path.join(IMAGE_ROOT, path), 'rb') as img:
            base64_string = base64.b64encode(img.read())

        headers = {'Content-Disposition': 'inline; filename="test.png"'}
        output_bytes.append(Response(base64_string, headers=headers, media_type='image/png'))
    
    #disease-plane importance 정보
    with open('./result.json', 'r') as file:
        result_dict = json.load(file)
    info = result_dict[disease][plane]

    result_info = {}
    #info + 이미지
    result_info['info'] = info
    result_info['img'] = output_bytes

    return result_info

# 질병 별 각 축의 가장 중요 슬라이드 + gradcam
@app.get("/result/{disease}/{method}")
async def resultFile(disease:str, method:str):
    if method == "original": #original or gradcam
        IMAGE_ROOT = os.path.join(disease, method) 
    elif method == "gradcam":
        IMAGE_ROOT = os.path.join(disease, method)
    
    output_bytes = []
    image_paths = os.listdir(IMAGE_ROOT)
    for path in image_paths:

        with open(os.path.join(IMAGE_ROOT, path), 'rb') as img:
            base64_string = base64.b64encode(img.read())

        headers = {'Content-Disposition': 'inline; filename="test.png"'}
        output_bytes.append(Response(base64_string, headers=headers, media_type='image/png'))

    return output_bytes