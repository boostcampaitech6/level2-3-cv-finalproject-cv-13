from fastapi import FastAPI, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
import os
import numpy as np
import base64


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

    for f in file:
        print(f.filename)
        image = Image.open(f.file)
        image.show()

    return {"uploadStatus": "Complete"}

@app.get("/outputoriginal")
async def outputFile():
    IMAGE_ROOT = 'sampleimages'
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