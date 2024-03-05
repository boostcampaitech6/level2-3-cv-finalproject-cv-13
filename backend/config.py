import torch
from pydantic import Field
from pydantic_settings import BaseSettings

class Config(BaseSettings):
    ext: list = [".dcm"]  # , ".png"
    planes: list = ["axial","coronal","sagittal"]
    diseases: list = ["abnormal", "acl", "meniscus"]
    orign_path: str = "original"
    result_path: str = "results"
    model_path: str = "models"
    model_class: str = "MRNet"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()