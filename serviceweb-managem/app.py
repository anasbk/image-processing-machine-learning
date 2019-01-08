from flask import Flask, request, jsonify

import cv2
import numpy as np
import scipy
from scipy.misc import imread
import pickle as pick
import random
import os
import matplotlib.pyplot as plt
import pandas as pd
import mahotas

import urllib
import numpy as np

bins = 8
# Feature extractor
#shape
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

#texture
def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick


#color
def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    return hist.flatten()
def average(lst):
    return sum(lst)/len(lst)
    
def extract_features(image_path, vector_size=32):
    image = imread(image_path, mode="RGB")
    result = []
    result.append(average(fd_hu_moments(image)))
    result.append(average(fd_haralick(image)))
    result.append(average(fd_histogram(image)))
    return result







app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"



@app.route("/data", methods=["POST"])
def api_request():
    
    filetosave = request.files['photo']

    filetosave.save('imageToSave.jpg')

    # load model

    filename = "linear_reg.model"

    loaded_model = pick.load(open("models/"+filename, 'rb'))
    
    print(loaded_model)
    

    # extract_features

    ft = extract_features('imageToSave.jpg')
    
    print(ft)


    rez = []
    rez.append(ft[1])
    rez.append(ft[0])
    rez.append(ft[2])

    print(np.array(ft).reshape(-1,1))

    print("ft")
    print(ft)
    print(rez)

 

    predicted_values = loaded_model.predict(np.array(rez).reshape(1,-1))

    print(predicted_values)

    resp = { 'pb':round(predicted_values[0,0],2),
             'cu':round(predicted_values[0,1],2),
             'zn':round(predicted_values[0,2],2)
             }
    
    print(resp)
  

    return jsonify(resp)
    

if __name__ == '__main__':
    app.run(host= '0.0.0.0', debug=True)

