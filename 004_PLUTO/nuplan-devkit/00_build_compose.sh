#!/bin/bash
set -e

source .env
SERVICE_NAME=simulation

docker compose build ${SERVICE_NAME}

sleep 0.5
echo "==========================================="
docker ps -a