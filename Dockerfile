FROM python:3.8-slim-buster

WORKDIR /data

COPY . /data

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8888

CMD ["python", "classification.py"]
