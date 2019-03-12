# https://medium.com/@sightengine_/image-upload-and-moderation-with-python-and-flask-e7585f43828a

from flask import Flask, render_template, request            
import subprocess 

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
from sklearn.externals import joblib

app = Flask(__name__)

algorithms = ['RetinaNet', 'SSD', 'MaskR-CNN', 'MultiPath Network', 'R-FCN', 'YOLO']

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

clf = joblib.load('/mnt/images/MSCOCO/result_val2014/svc.sav') # PATH TO PRE-TRAINED MODEL
model = models.alexnet(pretrained=True)
new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])

class DataSet(Dataset):        
    def __init__(self, filename):
        self.__xs = [filename]
        self.transform = transforms.Compose(
                                [transforms.Resize((227, 227)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])        
    def __getitem__(self, index):        
        img = Image.open(self.__xs[index])
        img = img.convert('RGB')
        img = self.transform(img)
        img = torch.from_numpy(np.asarray(img))
        fname = os.path.basename(os.path.normpath(self.__xs[index]))[:-4]
        return img, fname
    def __len__(self):
        return len(self.__xs)

def choose(fname, context):
    batch_size = 1
    data_set = DataSet(fname)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False, num_workers=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.classifier = new_classifier
    model.to(device)
    with torch.no_grad():
        for data in data_loader:
            images, fname = data
            images = images.to(device)
            outputs = model(images)
            outputs = outputs.view(batch_size, -1)
            for index, features in enumerate(outputs):
                features = features.cpu().numpy()   
    algo = clf.predict(np.hstack((features,context)).reshape(1, -1))[0]
    return algo

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    # ADD RESTRICTIONS TO FILE EXTENSIONS
    place = int(request.form['place'])
    inside = int(request.form['inside'])
    outside = int(request.form['outside'])
    filled = int(request.form['filled'])
    light = int(request.form['light'])
    surrounding = int(request.form['surrounding'])
    time = int(request.form['time'])
    action = int(request.form['action'])
    file = request.files['image'] 
    f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(f)
    algo = choose(os.path.join(app.config['UPLOAD_FOLDER'], file.filename), [place,inside,outside,filled,light,surrounding,time,action])
    return 'Best algorithm for this image is ' + algorithms[algo-1] + '.'

if __name__ == "__main__":
    app.run(debug=True)

