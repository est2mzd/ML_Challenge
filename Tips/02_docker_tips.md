# VS-Codeでコンテナ内に入る方法
```
1. F1 > Attach to Running Container で　コンテナを選択
2. Ctrl + Shit + p > Connect to Host
```
---
# common.sh　の例
```
# common.sh

# build.sh に必要な変数
IMAGE_NAME="pluto-model"
DOCKERFILE_DIR=$(dirname "$0")

# start.sh に必要な変数
CONTAINER_NAME="pluto-container"

# このファイルの1個上をマウントする
WORK_DIR=$(dirname "$(dirname "$0")")
```
---
# start.sh　の例
```
docker run \
    --gpus all \                            # GPUを使用
    --rm \                                  # コンテナ停止時に削除
    -d \                                    # デタッチモードで起動 ---> これだと、すぐに exitedになり rm　される
    -itd \                                  # デタッチモードで起動 ---> これが正解
    -v "$WORK_DIR":/work \                  # データディレクトリをマウント
    --name "$CONTAINER_NAME" \              # コンテナに名前を付ける
    --workdir /work \                       # 作業ディレクトリを設定
    --user $USERID \                        # ホストのユーザーIDを使用
    --env USERNAME=$USERNAME \              # ホストのユーザー名を環境変数として渡す
    --env USERID=$USERID \                  # ホストのユーザーIDを環境変数として渡す
    "$IMAGE_NAME"                           # 使用するイメージ
```
---
# build.sh　の例
```
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

```
---