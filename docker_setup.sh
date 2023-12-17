#!/usr/bin/env bash

image_name=snowglobe_image
container_name=snowglobe_container

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
       --env OPENAI_API_KEY=$(cat ~/src/llm/api/openai) \
       -v ~/src:/home/snowglobe/src \
       -v ~/wdata:/home/snowglobe/wdata \
       -v /nfs:/nfs \
       -v /local_data:/local_data \
       $image_name
