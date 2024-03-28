from fastapi import FastAPI, UploadFile, Response, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from PIL import Image
from typing import Optional

import time
import io
import os
import numpy as np
import base64
import json
import shutil
import warnings
from contextlib import asynccontextmanager
warnings.filterwarnings("ignore")

from utils import *
from schemas import DICOMRequest, resultResponse, DiseaseResult, PatientInfo
from dcm_convert import convert_dcm_to_numpy
from config import config
from auto_docs import summary_reports, SummaryReport


@asynccontextmanager
async def lifespan(app: FastAPI):
    s_time = time.time()
    inference_models.set_model()
    print("Model Loading Time: ", time.time() - s_time)
    yield
    
    
origins = [
    "http://localhost:3000",
    # "http://223.130.129.202:80",
    "http://223.130.130.192:80"
]

app = FastAPI(lifespan=lifespan)

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
async def receiveFile(plane:str, file: list[UploadFile], request: Request):
    id_root = request.headers['ip'][:6] # 6자리까지 사용
    print("post: ", id_root)

    UPLOAD_FOLDER = os.path.join(id_root, config.orign_path, plane)

    if os.path.exists(UPLOAD_FOLDER): 
        shutil.rmtree(UPLOAD_FOLDER) # 이미 폴더 있으면 삭제
    os.makedirs(UPLOAD_FOLDER)
    
    summary_reports[id_root] = SummaryReport()
    # DICOM 서버에 저장
    for f in file:
        ext = os.path.splitext(f.filename)[1]
        if ext.lower() not in config.ext:
            raise HTTPException(status_code=400, detail="지원되지 않는 파일 형식입니다.")

        try:
            file_path = os.path.join(UPLOAD_FOLDER, f.filename)
            with open(file_path, "wb") as file_object:
                file_object.write(f.file.read())

                info, npy_array = convert_dcm_to_numpy(file_path)
                np.save(os.path.join(UPLOAD_FOLDER, "input.npy"), npy_array)
                summary_reports[id_root].set_personal_info(info)

            return JSONResponse(status_code=200, content={ plane: "success"})
        except Exception as e:
            raise HTTPException(status_code=500, content={ plane: "파일 업로드에 실패했습니다.", "error": str(e)})

@app.get("/input/sample") # ex) /input/sample?disease=acl
async def receiveSampleFile(disease:str, request: Request):
    id_root = request.headers['ip'][:6]
    print("sample_post:", id_root)

    SAMPLE_PATH = os.path.join(config.sample_path, disease)
    UPLOAD_ROOT = os.path.join(id_root, config.orign_path)

    if os.path.exists(UPLOAD_ROOT): 
        shutil.rmtree(UPLOAD_ROOT) # 이미 폴더 있으면 삭제
    os.makedirs(UPLOAD_ROOT) # 새로 만들기

    summary_reports[id_root] = SummaryReport()
    # 데이터 복사
    shutil.copytree(
        SAMPLE_PATH,
        UPLOAD_ROOT,
        dirs_exist_ok=True
    )

    # numpy 생성
    for plane in config.planes:
        UPLOAD_PATH = os.path.join(UPLOAD_ROOT, plane)
        files = os.listdir(UPLOAD_PATH)
        for f in files:
            file_path = os.path.join(UPLOAD_PATH, f)
            info, npy_array = convert_dcm_to_numpy(file_path)
        np.save(os.path.join(UPLOAD_PATH, "input.npy"), npy_array)
    
    summary_reports[id_root].set_personal_info(info)
    return JSONResponse(status_code=200, content={ "sample" : "success"})
        

@app.get("/inference")
async def inference(request: Request):
    s_time = time.time()
    id_root = request.headers['ip'][:6]
    print("inference:", id_root)
    planes = config.planes
    diseases = config.diseases

    # JSON template 불러오기
    with open('./template.json', 'r') as file:
        result_dict = json.load(file)
    result_dict['percent']['labels'] = config.diseases

    for disease in diseases:
        res = []

        for plane in planes:
            print(f'Inference about {disease}-{plane}')

            input_path = os.path.join(id_root, config.orign_path, plane, 'input.npy')
            input_tensor = data_processing(input_path)
            res.append(predict_disease(input_tensor, disease, plane, "cuda"))
            
            max_idx, camscores = grad_cam_inference(id_root, input_tensor, disease, plane)

            datasets = [] 
            labels = []
            for i, score in enumerate(camscores):
                labels.append(i)
                datasets.append({"x": i, "y": round(score * 1000)})

            result_dict[disease][plane]['labels'] = labels
            result_dict[disease][plane]['datasets'] = datasets
            result_dict[disease][plane]['highest'] = max_idx

        proba = {}
        proba['y'] = disease
        fusion_res = predict_percent(res, disease)
        proba['x'] = round((fusion_res[0] * 100),1)
        result_dict['percent']['datasets'].append(proba)    
        
    # summary   
    cls_result = result_dict['percent']['datasets']
    _keys = [proba['y'] for proba in cls_result]
    _values = [proba['x'] for proba in cls_result]

    for k in _keys:
        _gradcam_path = os.path.join(id_root,"docs_img", k)
        summary_reports[id_root].set_image_paths(_gradcam_path)
    summary_reports[id_root].set_result_info(_values)

    with open(os.path.join(id_root,'result.json'),'w') as f:
        json.dump(result_dict, f, indent=4)

    print("Inference Time: ", time.time() - s_time)
    return {"inference" : "success"}

# 전체 결과
@app.get("/result")
async def outputJSON(request: Request) -> resultResponse:
    id_root = request.headers['ip'][:6]
    print("result: ",id_root)

    # JSON 파일 불러오기
    with open(os.path.join(id_root,'result.json'), 'r') as file:
        result_dict = json.load(file)
    result = result_dict['percent']
    return resultResponse(labels=result["labels"], datasets=result["datasets"])


@app.get("/result/patient")
async def patientInfo(request: Request) -> PatientInfo:
    id_root = request.headers['ip'][:6]
    patient_info = summary_reports[id_root].get_personal_info()
    return PatientInfo(labels=patient_info[0], info=patient_info[1])
    

# 질병 별 각 축의 가장 중요 슬라이드 + gradcam
@app.get("/result/{disease}/{method}") #?threshold=0.5
async def resultFile(request: Request, disease:str, method:str, threshold: Optional[float] = None):
    id_root = request.headers['ip'][:6]
    print("gradcam: ", id_root)

    ORIGIN_PATH = os.path.join(id_root, config.result_path, "original", disease)
    GRAD_PATH = os.path.join(id_root, config.result_path, "gradcam", disease)
    output_bytes = []
    numpy_files = sorted(os.listdir(ORIGIN_PATH))
    for f in numpy_files:
        original_img = np.load(os.path.join(ORIGIN_PATH, f))
        if method == "gradcam":
            if threshold == None:
                threshold = 0.5
            cam_img = np.load(os.path.join(GRAD_PATH, f))
            visualization = show_cam_on_image(original_img, cam_img, use_rgb=True, threshold=threshold)
            result_img = Image.fromarray(visualization)
        else:
            result_img = Image.fromarray((original_img*255).astype(np.uint8))

        byte_buffer = io.BytesIO()
        result_img.save(byte_buffer, format="PNG")

        base64_string = base64.b64encode(byte_buffer.getvalue())
        headers = {'Content-Disposition': 'inline; filename="test.png"'}
        output_bytes.append(Response(base64_string, headers=headers, media_type='image/png'))

    return output_bytes

# 원본이미지 + 질병에 따른 사진 별 score 그래프
@app.get("/output/{disease}/{plane}") # ex) ouput/abnormal/axial
async def outputFile(disease: str, plane:str, request: Request):
    id_root = request.headers['ip'][:6]
    print("gradcam: ", id_root)

    INPUT_ROOT = os.path.join(id_root, config.orign_path, plane)
    #이미지 정보
    output_bytes = []
    numpy_paths = os.path.join(INPUT_ROOT,'input.npy')

    npy_images = np.load(numpy_paths)
    for npy_img in npy_images:
        npy_img = Image.fromarray(npy_img)
        # PIL 이미지를 바이트 스트림으로 변환하여 메모리 버퍼에 저장
        byte_buffer = io.BytesIO()
        npy_img.save(byte_buffer, format="PNG")

        base64_string = base64.b64encode(byte_buffer.getvalue())
        headers = {'Content-Disposition': 'inline; filename="test.png"'}
        output_bytes.append(Response(base64_string, headers=headers, media_type='image/png'))
    
    #disease-plane importance 정보
    with open(os.path.join(id_root, 'result.json'), 'r') as file:
        result_dict = json.load(file)
    info = result_dict[disease][plane]

    result_info = {}
    #info + 이미지
    result_info['info'] = info
    result_info['img'] = output_bytes

    return result_info

@app.get("/result/docs")
async def exportSummary(request: Request, cache: str):
    id_root = request.headers['ip'][:6]
    print("docs: ", id_root)
    # need for summary report
    FILE_NAME = '123456_auto_report.docx'
    summary_reports[id_root].export_to_docx(id_root)
    response = FileResponse(os.path.join(id_root, FILE_NAME), media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document')
    response.headers["Cache-Control"] = "no-cache"
    return response