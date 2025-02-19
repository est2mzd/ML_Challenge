#!/bin/bash

CONTAINER_ID=64bf4757b22f
USER_ID=est2mzd
IMAGE_NAME=e2e_parking
TAG=v4

docker commit $CONTAINER_ID ${USER_ID}/${IMAGE_NAME}:${TAG}
