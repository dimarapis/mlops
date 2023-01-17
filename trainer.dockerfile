FROM python:3.9.1-slim-buster

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY src/ src/
COPY data/ data/
COPY setup.py setup.py

RUN pip install -r requirements.txt --no-cache-dir
#RUN pip uninstall -y nvidia-cublas-cu11

WORKDIR /

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
