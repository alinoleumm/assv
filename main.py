from flask import Flask, render_template            
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

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/algo")
def algorithm():
    # Provide a filename and context (front-end)
	algo = choose('/home/alinoleumm/images/42.jpg', [0]*8)
	return str(algo)

if __name__ == "__main__":
    app.run(debug=True)

