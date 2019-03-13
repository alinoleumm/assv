#!/bin/bash

sshpass -p password scp $1 lukacm@10.1.72.53:/home/lukacm/uploads/image.jpg
sshpass -p password ssh lukacm@10.1.72.53 "cd darknet; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64; ./darknet detect cfg/yolov3.cfg yolov3.weights data; rm /home/lukacm/uploads/*" > detection.txt
