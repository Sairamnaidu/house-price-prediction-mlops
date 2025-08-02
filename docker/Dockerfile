
FROM python:3.10-slim


WORKDIR /app


COPY ./requirements.txt /app/requirements.txt
COPY ./src /app/src
COPY ./mlruns /app/mlruns


RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000


CMD ["uvicorn", "src.predict_api:app", "--host", "0.0.0.0", "--port", "8000"]
