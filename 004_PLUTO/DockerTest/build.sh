#!/bin/bash

# common.shを読み込む
source "$(dirname "$0")/common.sh"

# ホストのユーザー名とユーザーIDを取得
USERNAME=$(whoami)
USERID=$(id -u)

# Dockerイメージをビルド
docker build \
    --build-arg USERNAME=$USERNAME \
    --build-arg USERID=$USERID \
    -t "$IMAGE_NAME" "$DOCKERFILE_DIR"
