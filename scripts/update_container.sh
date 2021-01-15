#!/usr/bin/env bash

cd /home/sam/Code/remote
docker build . -f gnosis/Dockerfile --tag samuelstanton/gnosis:py3.8_cuda11
docker push samuelstanton/gnosis:py3.8_cuda11