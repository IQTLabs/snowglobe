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
    ninja

# User account
ARG username=snowglobe
ARG groupname=$username
ARG uid=1000
ARG gid=$uid
RUN groupadd --gid $gid $groupname
RUN adduser --uid $uid --gid $gid --disabled-password $username
WORKDIR /home/$username

# Install
COPY --chown=$uid:$gid . /home/$username
RUN pip install cmake
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install .
USER $username
RUN snowglobe_config
