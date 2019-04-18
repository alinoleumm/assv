#!/bin/bash

sshpass -p pwd scp $1 root@94.156.144.96:/root/uploads/image.jpg

sshpass -p pwd ssh root@94.156.144.96 << EOF
  sshpass -p pwd scp -P 6666 /root/uploads/image.jpg lukacm@127.0.0.1:/home/lukacm/uploads/image.jpg
  sshpass -p pwd ssh -p 6666 lukacm@127.0.0.1
  1
  sshpass -p pwd scp /home/lukacm/uploads/image.jpg ayaz@10.1.73.196:/home/ayaz/uploads/image.jpg
  sshpass -p pwd ssh ayaz@10.1.73.196
  source /home/ayaz/anaconda3/bin/activate retina
  cd keras-retinanet
  python ResNet50RetinaNet.py
  sshpass -p pwd scp /home/ayaz/detections/detection.txt root@94.156.144.96:/root/detections/detection.txt
EOF

sshpass -p pwd scp root@94.156.144.96:/root/detections/detection.txt ./detection.txt