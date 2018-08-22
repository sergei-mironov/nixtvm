#!/usr/bin/env bash

# Uncomment to debug
set -x

while test -n "$1" ; do
  case "$1" in
    -h|--help)
      echo "Usage: $0 [--noport] [SUFFIX]" >&2
      exit 1
      ;;
    --noport)
      NOPORT=y
      ;;
    *)
      SUFFIX="$1"
      ;;
  esac
  shift
done

UID=`id --user`
test -z "$SUFFIX" && SUFFIX="dev"
test -z "$NOPORT" && NOPORT=n
test -z "$DOCKER_WORKSPACE" && DOCKER_WORKSPACE=`pwd`
test -z "$DOCKER_COMMAND" && DOCKER_COMMAND="/bin/bash"
DOCKER_CONTEXT_PATH="./tvm/docker"
DOCKERFILE_PATH="./Dockerfile.${SUFFIX}"

if echo "$SUFFIX" | grep -q gpu ; then
  CONTAINER_TYPE=gpu
  DOCKER_BINARY="nvidia-docker"
else
  CONTAINER_TYPE=cpu
  DOCKER_BINARY="docker"
fi

DOCKER_IMG_NAME=$(echo "hitvm.${CONTAINER_TYPE}" | sed -e 's/=/_/g' -e 's/,/-/g' | tr '[:upper:]' '[:lower:]')

if test -z "$DOCKER_PROXY_ARGS" ; then
  if test -n "$https_proxy" ; then
    PROXY_HOST=`echo $https_proxy | sed 's@.*//\(.*\):.*@\1@'`
    PROXY_PORT=`echo $https_proxy | sed 's@.*//.*:\(.*\)@\1@'`
    DOCKER_PROXY_ARGS="--build-arg=http_proxy=$https_proxy --build-arg=https_proxy=$https_proxy --build-arg=ftp_proxy=$https_proxy"
  fi
fi

# Copy additional files to context
cp -f Huawei.crt "$DOCKER_CONTEXT_PATH"
for f in _dist/* ; do
  echo "$DOCKER_CONTEXT_PATH/`basename $f` -> $f"
  ln $f $DOCKER_CONTEXT_PATH 2>/dev/null
done

docker build ${DOCKER_PROXY_ARGS} -t ${DOCKER_IMG_NAME} \
  -f "${DOCKERFILE_PATH}" "${DOCKER_CONTEXT_PATH}"

if [[ $? != "0" ]]; then
  echo "ERROR: docker build failed."
  exit 1
fi

# Remap detach to Ctrl+e,e
mkdir /tmp/docker-$UID || true
cat >/tmp/docker-$UID/config.json <<EOF
{ "detachKeys": "ctrl-e,e" }
EOF
DOCKER_CFG="--config /tmp/docker-$UID"

if test "$NOPORT" = "y"; then
  PORT_TENSORBOARD=`expr 6000 + $UID - 1000`
  PORT_JUPYTER=`expr 8000 + $UID - 1000`
  DOCKER_PORT_ARGS="-p 0.0.0.0:$PORT_TENSORBOARD:6006 -p 0.0.0.0:$PORT_JUPYTER:8888"
  echo
  echo "*****************************"
  echo "Your Jupyter port: ${PORT_JUPYTER}"
  echo "Your Tensorboard port: ${PORT_TENSORBOARD}"
  echo "*****************************"
fi

${DOCKER_BINARY} $DOCKER_CFG run --rm --pid=host \
  -v ${DOCKER_WORKSPACE}:/workspace \
  -w /workspace \
  -e "CI_BUILD_HOME=/workspace" \
  -e "CI_BUILD_USER=$(id -u -n)" \
  -e "CI_BUILD_UID=$(id -u)" \
  -e "CI_BUILD_GROUP=$(id -g -n)" \
  -e "CI_BUILD_GID=$(id -g)" \
  -e "MAVEN_OPTS=-Dhttps.proxyHost=$PROXY_HOST -Dhttps.proxyPort=$PROXY_PORT -Dmaven.wagon.http.ssl.insecure=true" \
  -e "DISPLAY=$DISPLAY" \
  -e "http_proxy=$http_proxy" \
  -e "https_proxy=$https_proxy" \
  ${DOCKER_PORT_ARGS} \
  -it \
  --cap-add SYS_PTRACE \
  ${DOCKER_IMG_NAME} \
  bash tvm/docker/with_the_same_user \
  ${DOCKER_COMMAND}

