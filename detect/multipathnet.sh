#!/bin/bash

sshpass -p pwd scp $1 lukacm@10.1.72.50:/home/lukacm/uploads/image.jpg
sshpass -p pwd ssh lukacm@10.1.72.50 "2; cd multipathnet/multipathnet-master; export CUDNN_PATH=\"/usr/local/cudnn-5/cuda/lib64/libcudnn.so.5\"; /home/lukacm/torch/install/bin/th demo.lua"
sshpass -p pwd scp lukacm@10.1.72.50:/home/lukacm/detections/detection.txt ./detection.txt