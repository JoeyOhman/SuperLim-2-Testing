#!/bin/bash
# echo "$(pwd)/.."
docker run -v $(pwd)/..:/workdir/ -p13377:13377 --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it pt_image:latest bash