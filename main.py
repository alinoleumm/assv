from flask import Flask, render_template, request  
from flask import send_file          
import subprocess 
import shlex
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
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import random
import itertools
import colorsys
from skimage.measure import regionprops
import scipy.spatial.distance as dist
from collections import namedtuple

app = Flask(__name__)

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

catNames = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic_light',
               'fire_hydrant', 'stop_sign', 'parking_meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball',
               'kite', 'baseball_bat', 'baseball_glove', 'skateboard',
               'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted_plant', 'bed',
               'dining_table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell_phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy_bear', 'hair_drier', 'toothbrush']

catIds = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,
             27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,
             50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,
             74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90]

def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

algorithms = ['RetinaNet', 'SSD', 'MaskR-CNN', 'MultiPath Network', 'R-FCN', 'YOLO']
cat_dir = '/home/alinoleumm/assv_assets/object_histograms/' # PATH TO OBJECT HISTOGRAMS

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

wclf = joblib.load('/home/alinoleumm/assv_assets/svc.sav') # PATH TO PRE-TRAINED MODEL FOR WHOLE IMAGE
iclf = joblib.load('/home/alinoleumm/assv_assets/dt.sav') # PATH TO PRE-TRAINED MODEL FOR INSTANCES
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

def removeContradiction(fname, instid):
    lines = [line.rstrip('\n') for line in open('detection.txt')]
    file = open('detection_2.txt', 'w')
    for n in range(int(len(lines)/7)):
        if n!=instid:
            for i in range(7):
                print(lines[7*n+i])
                file.write(lines[7*n+i] + '\n')
        else:
            xmin = int(lines[7*n])
            ymin = int(lines[7*n+1])
            xmax = int(lines[7*n+2])
            ymax = int(lines[7*n+3])
    file.close()
    img = Image.open(fname)
    width, height = img.size
    Color = namedtuple("Color", "R G B")
    blackColor = Color(0x00, 0x00, 0x00)
    for w in range(width):
        for h in range(height):
            if w >= xmin and w <= xmax and h >= ymin and h <= ymax:
                continue
            else:
                img.putpixel((w,h), blackColor)
    img.save('uploads_2/' + fname.split('/')[-1])
    return xmin, ymin, xmax, ymax


def chooseWhole(fname, context):
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
    algo = wclf.predict(np.hstack((features,context)).reshape(1, -1))[0]
    return algo

def norm(data):
    data_norm = (data-min(data))/(max(data)-min(data))
    weights = np.ones_like(data_norm)/float(len(data_norm))
    hist = np.histogram(data_norm, weights=weights) 
    return hist[0]

def chooseInstance(fname, xmin, ymin, xmax, ymax, gt): 

    img = Image.open(fname).convert('L')
    width, height = img.size
    obj = img.crop((xmin, ymin, xmax, ymax))
    # obj.save('couch.png')
    objWidth, objHeight = obj.size
    regions = regionprops(obj)

    obj_hist = []
    data = sorted([r['area']/(width*height) for r in regions])[-50:]   
    obj_hist = np.hstack((obj_hist, norm(data)))
    centroidx = sorted([r['centroid'][0]/(width) for r in regions])[-50:]  
    obj_hist = np.hstack((obj_hist, norm(data)))
    centroidy = sorted([r['centroid'][1]/(height) for r in regions])[-50:]  
    obj_hist = np.hstack((obj_hist, norm(data)))
    eccentricities = sorted([r['eccentricity'] for r in regions])[-50:]  
    obj_hist = np.hstack((obj_hist, norm(data)))
    equivalent_diameters = sorted([r['equivalent_diameter']/r['area'] for r in regions])[-50:]   
    obj_hist = np.hstack((obj_hist, norm(data)))
    extents = sorted([r['extent'] for r in regions])[-50:]  
    obj_hist = np.hstack((obj_hist, norm(data)))
    orientations = sorted([r['orientation'] for r in regions])[-50:]  
    obj_hist = np.hstack((obj_hist, norm(data)))
    perimeters = sorted([r['perimeter']/((width+height)*2) for r in regions])[-50:]  
    obj_hist = np.hstack((obj_hist, norm(data)))
    solidities = sorted([r['solidity'] for r in regions])[-50:]  
    obj_hist = np.hstack((obj_hist, norm(data)))

    sims = np.empty((0,7))
        
    for cl in catNames:
        sim = []
        obj = np.load(cat_dir + cl + '.npy')
        sim.append(dist.braycurtis(obj_hist,obj))
        sim.append(dist.canberra(obj_hist,obj))
        sim.append(dist.cityblock(obj_hist,obj))
        sim.append(dist.chebyshev(obj_hist,obj))
        sim.append(dist.correlation(obj_hist,obj))
        sim.append(dist.cosine(obj_hist,obj))
        sim.append(dist.euclidean(obj_hist,obj))
        sims = np.append(sims, np.expand_dims(sim, axis=0), axis=0)

    gtsim = [gt]
    for i in range(7):
        gtsim.append(catIds[sims[:,i].tolist().index(min(sims[:,i]))])

    algo = iclf.predict(gtsim)[0]
    return algo

def detect(fname, algo, isinst):

    if algo==1:
        subprocess.call(shlex.split('./detect/retinanet.sh ' + fname))
    elif algo==2:
        subprocess.call(shlex.split('./detect/ssd.sh ' + fname))
    elif algo==3:
        subprocess.call(shlex.split('./detect/maskrcnn.sh ' + fname))
    elif algo==4:
        subprocess.call(shlex.split('./detect/multipathnet.sh ' + fname))
    elif algo==5:
        subprocess.call(shlex.split('./detect/rfcn.sh ' + fname))
    else:
        subprocess.call(shlex.split('./detect/yolo.sh ' + fname))

    if isinst:
        detall = open('detection.txt', 'a+')
        lines = [line.rstrip('\n') for line in open('detection_2.txt')]
        for line in lines:
            detall.write(line + '\n')
        detall.close()

    image = skimage.io.imread('/home/alinoleumm/assv/uploads/' + fname.split('/')[-1])
    imgw = image.shape[0]
    imgh = image.shape[1]

    lines = [line.rstrip('\n') for line in open('detection.txt')]
    numinst = int(len(lines)/7)
                                   
    instances = np.empty((0,6))
    for n in range(numinst):
        inst = []
        x = float("{0:.2f}".format(float(lines[7*n])))
        y = float("{0:.2f}".format(float(lines[7*n+1])))
        width = float("{0:.2f}".format(float(lines[7*n+2])))
        height = float("{0:.2f}".format(float(lines[7*n+3])))
        category_id = int(lines[7*n+4])
        alg = int(lines[7*n+6])
        inst.append(x)
        inst.append(y)
        inst.append(width)
        inst.append(height)
        inst.append(category_id)
        inst.append(alg)
        instances = np.append(instances, np.expand_dims(inst, axis=0), axis=0)    

    colors = random_colors(instances.shape[0])

    plt.rcParams['figure.figsize'] = (imgh/50, imgw/50)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    plt.imshow(image)
    currentAxis = plt.gca()
    currentAxis.axis('off')

    for i in range(len(instances)):
        color = colors[i]
        p = instances[i]
        coords = (p[0], p[1]), p[2]-p[0]+1, p[3]-p[1]+1
        display_txt = class_names[(int(p[4]))] + '\n' + str(algorithms[int(p[5])])
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(p[0], p[1], display_txt, fontsize=13, bbox={'facecolor':color, 'alpha':0.5})

    if isinst:
        detfname = 'static/detections/' + fname.split('/')[-1].split('.')[0] + '_' + str(algorithms[algo-1]) + '_stage_2.jpg'
    else:
        detfname = 'static/detections/' + fname.split('/')[-1].split('.')[0] + '_' + str(algorithms[algo-1]) + '_stage_1.jpg'

    plt.savefig(detfname)
    plt.close()

    return detfname

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/upload', methods=['POST'])
def upload_file():
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
    algowhole = chooseWhole('/home/alinoleumm/assv/uploads/' + str(file.filename), [place,inside,outside,filled,light,surrounding,time,action])   
    detfname = detect('/home/alinoleumm/assv/uploads/' + str(file.filename), algowhole, False)
    '''
    SEMANTIC VERIFICATION 
    should return contradicting detection and hypothesis
    '''
    # xmin, ymin, xmax, ymax = removeContradiction('/home/alinoleumm/assv/uploads/' + str(file.filename), 0)
    # algoinst = chooseInstance('/home/alinoleumm/assv/uploads/' + str(file.filename), xmin, ymin, xmax, ymax, -1) 
    # detfname = detect('/home/alinoleumm/assv/uploads_2/' + str(file.filename), algoinst, True)
    return render_template('home.html', filename=detfname)

if __name__ == "__main__":
    app.run(debug=True)

