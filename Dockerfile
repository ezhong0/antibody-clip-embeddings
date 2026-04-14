FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
  # trigger rebuild for auto-publish test4                                                                       
COPY . /app

RUN mkdir -p inputs out
