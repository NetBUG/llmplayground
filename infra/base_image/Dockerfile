FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04 AS py311-base

ARG VAULT_CA_URI=${VAULT_CA_URI:-http://vault.lan:8200/v1/pki/ca/pem}

WORKDIR /app

ENV HF_HOME=/app/cache

USER root
# add portaudio19-dev if needed for voice IO
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata \
    && apt-get --assume-yes install build-essential cmake curl git libpng-dev libjpeg-dev ffmpeg software-properties-common \
    && apt-get update \
    && apt-get install -y cuda-nvcc-12-4 cuda-cudart-12-4 cuda-cudart-dev-12-4 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && DEBIAN_FRONTEND=noninteractive apt install -y python3.11 python3.11-dev \
    && curl https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/.aws \
    && curl -o /usr/local/share/ca-certificates/ca.crt ${VAULT_CA_URI} \
    && update-ca-certificates

FROM py311-base

ARG TRANSFORMERS_DEPLOY_TOKEN
ARG TRANSFORMERS_DEPLOY_TOKEN_PASSWORD

WORKDIR /app

ENV HF_HOME=/app/cache

USER root

RUN pip3 install --upgrade cython pip \
    && python3.11 -m pip install --upgrade torch==2.4.0 torchvision torchaudio torchdata --index-url https://download.pytorch.org/whl/cu124 \
    && python3.11 -m pip cache purge

COPY pip.conf /etc/pip.conf
RUN pip config -v list
