FROM nvidia/cuda:12.2.2-devel-ubuntu22.04
LABEL org.iqt.name="snowglobe"

RUN apt update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    emacs \
    less \
    tree \
    wget \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

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
USER root
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install .
USER $username
RUN snowglobe_config
