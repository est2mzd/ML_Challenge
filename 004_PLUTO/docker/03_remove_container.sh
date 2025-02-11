#!/bin/bash
set -e

docker container rm nuplan_sim_${USER}
docker container rm nuplan_sub_${USER}

sleep 0.5
echo "==========================================="
docker ps -a