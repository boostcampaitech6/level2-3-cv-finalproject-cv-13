FROM python:3.10-slim as requirements-stage

WORKDIR /tmp
RUN pip install poetry
COPY ./pyproject.toml ./poetry.lock /tmp/
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

FROM python:3.10-slim as build-stage

WORKDIR /app
RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx libglib2.0-0
COPY --from=requirements-stage /tmp/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
COPY . /app

ENV PORT=8000

CMD  ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]