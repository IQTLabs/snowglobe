#!/usr/bin/env bash

image_name=llm_image
container_name=llm_container

if [ -z "$(docker images -q $image_name)" ]; then
    docker build -f Dockerfile -t $image_name \
	   --build-arg uid=$(id -u) \
	   --build-arg gid=$(id -g) \
	   --label "org.iqtlabs.user=$USER" \
	   ./
fi

docker run --name $container_name \
       --runtime=nvidia \
       -it --ipc=host --shm-size=64g \
       --gpus all \
       --device /dev/nvidiactl \
       --device /dev/nvidia0 \
       --device /dev/nvidia1 \
       --device /dev/nvidia2 \
       -v ~/src:/home/llm/src \
       -v ~/wdata:/home/llm/wdata \
       -v /nfs:/nfs \
       -v /local_data:/local_data \
       $image_name
