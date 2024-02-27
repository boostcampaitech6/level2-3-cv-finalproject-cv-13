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

