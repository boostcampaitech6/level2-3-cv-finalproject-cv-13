import torch
from pydantic import Field
from pydantic_settings import BaseSettings

class Config(BaseSettings):
    ext: list = [".dcm"]  # , ".png"
    planes: list = ["axial","coronal","sagittal"]
    diseases: list = ["abnormal", "acl", "meniscus"]
    model_class: str = "mrnet"
    orign_path: str = "./original"
    result_path: str = "./result"
    model_path: str = "./models"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()