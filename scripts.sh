#!/bin/sh
docker_image=test/yolov5:cuda11.7-runtime
# build
docker build -t ${docker_image} -f Dockerfile .

# run container
docker run -it --rm --name yolov5 --gpus all -p 5000:5000 ${docker_image}