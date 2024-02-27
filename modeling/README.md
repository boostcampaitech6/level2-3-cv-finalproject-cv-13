## Environment Settings

### Install Poetry

```bash
# 아래 설치 안되면 / 폴더 수정 권한 없으면
sudo su -
chmod -R 777 level2-3-cv-finalproject-cv-13/

apt install python3-pip
pip install poetry
poetry # 아무것도 안뜨면 안됨.
```

### Install Library

```bash
poetry init
poetry config virtualenvs.in-project true --local
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
poetry run python train.py
poetry run python inference.py
```

