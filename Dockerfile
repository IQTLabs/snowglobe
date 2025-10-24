FROM nvidia/cuda:12.6.3-devel-ubuntu24.04
LABEL org.iqt.name="snowglobe"

RUN apt update && apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    vim \
    less \
    wget \
    cmake \
    git \
    ninja-build

# User account
ARG username=snowglobe
ARG groupname=$username
ARG uid=1000
ARG gid=$uid
RUN deluser --remove-home ubuntu && \
    addgroup --gid $gid $groupname && \
    adduser --uid $uid --gid $gid --disabled-password $username && \
    mkdir /home/$username/logs && \
    chown $uid:$gid /home/$username/logs
WORKDIR /home/$username
USER $username

# Install
COPY --chown=$uid:$gid . /home/$username
RUN python3 -m venv /home/$username/.venv
ENV PATH=/home/$username/.venv/bin:"$PATH"
RUN pip install cmake && \
    CMAKE_ARGS="-DGGML_CUDA=on" pip install .
RUN snowglobe_config
