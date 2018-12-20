#!/bin/bash

# Set image name
IMAGE="cityscapes:latest"
if [ $# -eq 1 ]; then
    IMAGE=$1
fi

# Set project root dicrectory to map to docker 
THIS_DIR=$(cd $(dirname $0); pwd)
PROJ_DIR=`dirname ${THIS_DIR}`

# Make some directories if not exist
#mkdir -p ${PROJ_DIR}/data ${PROJ_DIR}/models

# Run container
CONTAINER="cityscapes"

nvidia-docker run -it --rm --ipc=host \
	-p 8889:8888 -p 6007:6006 \
	-v ${PROJ_DIR}:/work \
	--name ${CONTAINER} \
	${IMAGE}