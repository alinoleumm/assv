#!/bin/bash

sshpass -p pwd scp $1 ayaz@10.1.73.196:/home/ayaz/uploads/image.jpg
sshpass -p pwd ssh ayaz@10.1.73.196 "source /home/ayaz/anaconda3/bin/activate retina; cd keras-retinanet; python ResNet50RetinaNet.py"
sshpass -p pwd scp ayaz@10.1.73.196:/home/ayaz/detections/detection.txt detection.txt