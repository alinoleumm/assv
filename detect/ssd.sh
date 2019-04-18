#!/bin/bash

sshpass -p pwd scp $1 root@94.156.144.96:/root/uploads/image.jpg

sshpass -p pwd ssh root@94.156.144.96 << EOF
  sshpass -p pwd scp -P 6666 /root/uploads/image.jpg lukacm@127.0.0.1:/home/lukacm/uploads/image.jpg
  sshpass -p pwd ssh -p 6666 lukacm@127.0.0.1
  1
  cd caffe/caffe/examples
  python ssd_detect.py
  sshpass -p pwd scp /home/lukacm/detections/detection.txt root@94.156.144.96:/root/detections/detection.txt
EOF

sshpass -p pwd scp root@94.156.144.96:/root/detections/detection.txt ./detection.txt