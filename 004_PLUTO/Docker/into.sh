#!/bin/bash

# common.shを読み込む
source "$(dirname "$0")/common.sh"

# Dockerコンテナに接続する
docker exec -it "$CONTAINER_NAME" /bin/bash