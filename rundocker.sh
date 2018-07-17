#!/usr/bin/env bash

# Uncomment to debug
# set -x

UID=`id --user`
DOCKER_CONTEXT_PATH="./tvm/docker"
CI_DOCKER_EXTRA_PARAMS+=('-it')
CONTAINER_TYPE="dev"
DOCKERFILE_PATH="./Dockerfile.${CONTAINER_TYPE}"
COMMAND="/bin/bash"
PORT_JUPYTER=`expr 8000 + $UID - 1000`
PORT_TENSORBOARD=`expr 6000 + $UID - 1000`
RM="" # Set "--rm" to remove image after use

# Use nvidia-docker if the container is GPU.
if [[ "${CONTAINER_TYPE}" == *"gpu"* ]]; then
    DOCKER_BINARY="nvidia-docker"
else
    DOCKER_BINARY="docker"
fi

# Set up WORKSPACE and BUILD_TAG. Jenkins will set them for you or we pick
# reasonable defaults if you run it outside of Jenkins.
WORKSPACE="${WORKSPACE:-`pwd`}"
BUILD_TAG="${BUILD_TAG:-tvm-ci}"

# Determine the docker image name
DOCKER_IMG_NAME="${BUILD_TAG}.${CONTAINER_TYPE}"

# Under Jenkins matrix build, the build tag may contain characters such as
# commas (,) and equal signs (=), which are not valid inside docker image names.
DOCKER_IMG_NAME=$(echo "${DOCKER_IMG_NAME}" | sed -e 's/=/_/g' -e 's/,/-/g')

# Convert to all lower-case, as per requirement of Docker image names
DOCKER_IMG_NAME=$(echo "${DOCKER_IMG_NAME}" | tr '[:upper:]' '[:lower:]')

# HACK: setup a proxy
if test -n "$https_proxy" ; then
  all_proxy="http://10.122.85.159:3128"
  PROXY="--build-arg=http_proxy=$all_proxy --build-arg=https_proxy=$all_proxy --build-arg=ftp_proxy=$all_proxy"
else
  PROXY=""
fi


# HACK: Remap detach to Ctrl+e,e
mkdir /tmp/docker-$UID || true
cat >/tmp/docker-$UID/config.json <<EOF
{ "detachKeys": "ctrl-e,e" }
EOF
CFG="--config /tmp/docker-$UID"

# Build the docker container.
echo "Building container (${DOCKER_IMG_NAME})..."
docker build $PROXY -t ${DOCKER_IMG_NAME} \
    -f "${DOCKERFILE_PATH}" "${DOCKER_CONTEXT_PATH}"

# Check docker build status
if [[ $? != "0" ]]; then
    echo "ERROR: docker build failed."
    exit 1
fi

echo "PROXY: ${PROXY}"
echo "WORKSPACE: ${WORKSPACE}"
echo "CI_DOCKER_EXTRA_PARAMS: ${CI_DOCKER_EXTRA_PARAMS[@]}"
echo "COMMAND: ${COMMAND[@]}"
echo "CONTAINER_TYPE: ${CONTAINER_TYPE}"
echo "BUILD_TAG: ${BUILD_TAG}"
echo "DOCKER_IMG_NAME: ${DOCKER_IMG_NAME}"
echo
echo "*****************************"
echo "Your Jupyter port: ${PORT_JUPYTER}"
echo "Your Tensorboard port: ${PORT_TENSORBOARD}"
echo "*****************************"

${DOCKER_BINARY} $CFG run $RM --pid=host \
    -v ${WORKSPACE}:/workspace \
    -w /workspace \
    -e "CI_BUILD_HOME=/workspace" \
    -e "CI_BUILD_USER=$(id -u -n)" \
    -e "CI_BUILD_UID=$(id -u)" \
    -e "CI_BUILD_GROUP=$(id -g -n)" \
    -e "CI_BUILD_GID=$(id -g)" \
    -e "DISPLAY=$DISPLAY" \
    -e "http_proxy=$http_proxy" \
    -e "https_proxy=$https_proxy" \
    -p 0.0.0.0:$PORT_TENSORBOARD:6006 \
    -p 0.0.0.0:$PORT_JUPYTER:8888 \
    ${CI_DOCKER_EXTRA_PARAMS[@]} \
    ${DOCKER_IMG_NAME} \
    bash tvm/docker/with_the_same_user \
    ${COMMAND[@]}

