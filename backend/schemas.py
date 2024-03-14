from pydantic import BaseModel
from fastapi import UploadFile, Response

class DICOMRequest(BaseModel):
    plane: str
    #plane에 [axial, coronal, sagittal]만 가능한 것도 추가 해야하나...?
    # File : list[UploadFile]

class PatientInfo(BaseModel):
    labels: list[str]
    info: list[str]
   
class resultResponse(BaseModel):
    labels: list[str]
    datasets: list[dict]

class PlaneResult(BaseModel):
    labels: list[str]
    datasets: list[dict]
    highteset: int

class DiseaseResult(BaseModel):
    disease: str
    axial: PlaneResult
    coronal: PlaneResult
    sagittal: PlaneResult

# "abnormal": {
#         "axial": {
#             "labels": [
#                 0,
#                 1,
#                 2,
#                 3,
#                 4,
#                 5,
#                 6,
#                 7,
#                 8,
#                 9
#             ],
#             "datasets": [
#                 {
#                     "x": 0,
#                     "y": 50
#                 },
#                 {
#                     "x": 1,
#                     "y": 45
#                 },
#                 {
#                     "x": 2,
#                     "y": 15
#                 },
#                 {
#                     "x": 3,
#                     "y": 70
#                 },
#                 {
#                     "x": 4,
#                     "y": 98
#                 },
#                 {
#                     "x": 5,
#                     "y": 79
#                 },
#                 {
#                     "x": 6,
#                     "y": 24
#                 },
#                 {
#                     "x": 7,
#                     "y": 43
#                 },
#                 {
#                     "x": 8,
#                     "y": 67
#                 },
#                 {
#                     "x": 9,
#                     "y": 23
#                 }
#             ],
#             "highest": 4
#         },
    
