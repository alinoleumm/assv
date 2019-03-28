#!/bin/bash

sshpass -p pwd scp $1 lukacm@10.1.72.50:/home/lukacm/uploads/image.jpg
sshpass -p pwd ssh lukacm@10.1.72.50 "1; cd caffe/caffe/examples; python ssd_detect.py"
sshpass -p pwd scp lukacm@10.1.72.50:/home/lukacm/detections/detection.txt ./detection.txt