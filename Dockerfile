FROM nvidia/cuda:12.2.0-devel-ubuntu20.04
LABEL org.iqtlabs.name Scenario

RUN apt update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt install -y \
    python3 \
    python3-pip \
    emacs \
    less \
    tree \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install \
    numpy \
    torch \
    torchvision \
    transformers \
    einops \
    accelerate \
    tqdm \
    langchain \
    openai

# User account
ARG username=scenario
ARG groupname=$username
ARG uid=1000
ARG gid=$uid
RUN groupadd --gid $gid $groupname
RUN adduser --uid $uid --gid $gid --disabled-password $username
USER $username
WORKDIR /home/$username
