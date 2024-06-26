FROM nvidia/cuda:12.2.2-devel-ubuntu22.04
LABEL org.iqtlabs.name snowglobe

RUN apt update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt install -y \
    python3 \
    python3-pip \
    emacs \
    less \
    tree \
    wget \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip3 install \
    numpy \
    torch \
    torchvision \
    transformers \
    einops \
    accelerate \
    tqdm \
    pyyaml \
    fastapi[all] \
    langchain \
    openai \
    langchain-openai \
    langchain-huggingface \
    langchain-chroma \
    langchain-community \
    llama-cpp-python \
    poetry
ENV LLAMA_CPP_LIB=/usr/local/lib/python3.10/dist-packages/llama_cpp/libllama.so

# User account
ARG username=snowglobe
ARG groupname=$username
ARG uid=1000
ARG gid=$uid
RUN groupadd --gid $gid $groupname
RUN adduser --uid $uid --gid $gid --disabled-password $username
USER $username
WORKDIR /home/$username

# Install
COPY --chown=$uid:$gid . /home/$username
RUN pip install -e .
RUN ./download.sh
