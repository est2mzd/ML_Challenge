# common.sh

# build.sh に必要な変数
IMAGE_NAME="pluto-model"
DOCKERFILE_DIR=$(dirname "$0")

# start.sh に必要な変数
CONTAINER_NAME="pluto-container"

# このファイルの1個上をマウントする
#WORK_DIR=$(dirname "$(dirname "$0")")
WORK_DIR=$(dirname "$(dirname "$(realpath "$0")")")



