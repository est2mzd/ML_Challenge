#!/bin/bash
set -e

docker container stop nuplan_sim_${USER}
docker container stop nuplan_sub_${USER}

sleep 0.5
echo "==========================================="
docker ps -a