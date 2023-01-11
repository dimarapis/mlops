FROM python:3.7-slim
FROM  nvcr.io/nvidia/pytorch:22.07-py3

WORKDIR /
COPY cuda_availability.py cuda_availability.py
#RUN python cuda_availability.py
ENTRYPOINT ["python", "-u", "cuda_availability.py"]

