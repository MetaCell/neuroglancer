#!/bin/bash -xve

# This script builds `jxl_decoder.wasm` using emsdk in a docker container.

cd "$(dirname "$0")"

docker build .
docker run \
       --rm \
       -v ${PWD}:/src \
       -u $(id -u):$(id -g) \
       $(docker build -q .) \
       /src/build_wasm.sh
