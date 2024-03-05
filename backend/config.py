from pydantic import Field
from pydantic_settings import BaseSettings
import os

class Config(BaseSettings):
    ext: list = [".dcm", ".png"]
    planes: list = ["axial","coronal","sagittal"]
    tasks: list = ["abnormal", "acl", "meniscus"]
    orign_path: str = "original"
    result_path: str = "results"
    model_path: str = "models"

config = Config()