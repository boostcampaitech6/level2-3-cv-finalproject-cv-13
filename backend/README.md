### Install Library

```bash
# 해당 작업은 /modeling/ 혹은 /backend/ 에서 진행해주세요
pyenv local 3.10.0

poetry init
poetry config virtualenvs.in-project true --local
poetry env use python3.10
poetry run python --version
poetry install
```

### Install pytorch with poetry

```bash
poetry source add -p explicit pytorch https://download.pytorch.org/whl/cu117
poetry add --source pytorch torch==2.0.1+cu117 torchvision==0.15.2+cu117
poetry run python -c "import torch;print(torch.cuda.is_available())"  ## check pytorch installed
```

### Add new library

```bash
poetry add <library_name> #  = pip install <library_name>
```

### Train / Inference

```bash
poetry run uvicorn main:app --reload
```

### Folder
backend를 실행하기 위해 사전에 존재하는 파일 구조입니다.
```
📦level2-3-cv-finalproject-cv-13
┣ 📂 backend
┃ ┣ 📂 models
┃ ┃ ┣ 📜 abnormal_axial_best.pth
┃ ┃ ┣ 📜 abnormal_coronal_best.pth
┃ ┃ ┣ 📜 abnormal_sagittal_best.pth
┃ ┃ ┣ 📜 acl_axial_best.pth
┃ ┃ ┣ 📜 acl_coronal_best.pth
┃ ┃ ┣ 📜 acl_sagittal_best.pth
┃ ┃ ┣ 📜 meniscus_axial_best.pth
┃ ┃ ┣ 📜 meniscus_coronal_best.pth
┃ ┃ ┣ 📜 meniscus_sagittal_best.pth
┃ ┃ ┣ 📜 lr_abnormal.pkl
┃ ┃ ┣ 📜 lr_acl.pkl
┃ ┃ ┗ 📜 lr_meniscus.pkl
┃ ┣ 📂 pytorch_grad_cam 
┃ ┣ 📜 config.py
┃ ┣ 📜 dcm_convert.py
┃ ┣ 📜 main.py
┃ ┣ 📜 model.py
┃ ┣ 📜 schemas.py
┃ ┣ 📜 utils.py
┃ ┣ 📜 template.json
┗ 📂 ...
```