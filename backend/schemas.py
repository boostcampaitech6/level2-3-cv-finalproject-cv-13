from pydantic import BaseModel
from fastapi import UploadFile, Response

class DICOMRequest(BaseModel):
    plane: str
    #plane에 [axial, coronal, sagittal]만 가능한 것도 추가 해야하나...?
    File : list[UploadFile]

