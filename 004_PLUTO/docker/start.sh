#!/bin/bash

# common.shを読み込む
source "$(dirname "$0")/common.sh"

# ホストのユーザー名とユーザーIDを取得
USERNAME=$(whoami)
USERID=$(id -u)

# Dockerコンテナをデタッチモードで起動
docker run \
    --gpus all \
    --rm \
    -itd \
    -v "$WORK_DIR":/work \
    --name "$CONTAINER_NAME" \
    --workdir /work \
    --user $USERID \
    --env USERNAME=$USERNAME \
    --env USERID=$USERID \
    "$IMAGE_NAME"