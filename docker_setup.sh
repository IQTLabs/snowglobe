#!/usr/bin/env bash

image_name=snowglobe
container_name=snowglobe

if [ -z "$(docker images -q $image_name)" ]; then
    docker build -f Dockerfile -t $image_name \
	   --build-arg uid=$(id -u) \
	   --build-arg gid=$(id -g) \
	   --label "org.iqt.user=$USER" \
	   ./
fi

docker run --name $container_name \
       --runtime=nvidia \
       -it --shm-size=64g -p 8000:8000 \
       $image_name
