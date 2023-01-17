FROM python:3.9.1-slim-buster

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# install git
RUN apt-get -y update
RUN apt-get -y install git

RUN git clone https://github.com/dimarapis/mlops.git mlops/
WORKDIR /mlops

RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["scripts/trainer.sh"]
