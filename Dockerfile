FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
  # trigger rebuild for auto-publish test4                                                                       
COPY . /app

RUN mkdir -p inputs out
# bump to force rebuild 1776383376
# trigger rebuild post-refactor 1776387187
# rebuild again for the refactored backend 1776387752
# rebuild for 200MB test 1776390269
