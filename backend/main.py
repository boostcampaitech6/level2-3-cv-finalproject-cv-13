from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

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
async def receiveFile(file: list[UploadFile], id: int = Form(...)):

    for f in file:
        print(f.filename)
        image = Image.open(f.file)
        image.show()
        print(id)

    return {"uploadStatus": "Complete",
            "id": id}