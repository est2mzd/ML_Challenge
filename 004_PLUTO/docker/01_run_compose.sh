#!/bin/bash
set -e

source .env
SERVICE_NAME=simulation

# サービス名を指定して、docker imageを作成する場合
docker compose up ${SERVICE_NAME} -d

sleep 0.5
echo "==========================================="
docker ps -a