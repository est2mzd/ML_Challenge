#!/bin/bash

source ./common.sh

docker run \
    --gpus all \
    --net=host \
    -itd \
    --shm-size=8G \
    -v ${PARENT_DIR}:/home/$USER_NAME \
    --name $CONTAINER_NAME \
    $IMAGE_NAME
