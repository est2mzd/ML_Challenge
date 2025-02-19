#!/bin/bash

# build.sh に必要な変数
IMAGE_NAME=parking_sha_nmpc

# コンテナ名
CONTAINER_NAME=${IMAGE_NAME}_${USER}

# フルパスの取得
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$CURRENT_DIR")"
GRANDPARENT_DIR="$(dirname "$PARENT_DIR")"

# ホストのユーザー情報
USER_NAME=$(whoami)
USER_ID=$(id -u) #USER_ID=1001
USER_GID=$(id -g)

# コンテナの作業ディレクトリ
WORK_DIR=${PARENT_DIR}

echo "---------- common.sh ------------"
echo "Image Name     = ${IMAGE_NAME}"
echo "Container Name = ${CONTAINER_NAME}"
echo "WORK_DIR       = ${WORK_DIR}"
echo "USER_NAME  = ${USER_NAME}"
echo "USER_ID    = ${USER_ID}"
echo "USER_GID   = ${USER_GID}"
echo "---------------------------------"