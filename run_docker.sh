#!/usr/bin/env bash

docker run --rm -it --runtime=nvidia \
    -p 8888:8888 \
    -v ${PWD}:/ppf-registration-spatial-hashing \
    -h ppf-registration-spatial-hashing ppf-registration-spatial-hashing
