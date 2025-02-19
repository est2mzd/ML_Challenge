#!/bin/bash

FILE_PATH=e2e_parking_v4.tar
USER_ID=est2mzd
IMAGE_NAME=e2e_parking
TAG=v4

docker save -o ${FILE_PATH} ${USER_ID}/${IMAGE_NAME}:${TAG}
