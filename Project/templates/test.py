from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os, re, joblib
import numpy as np
import pandas as pd
from PIL import Image
from konlpy.tag import Okt
from tensorflow import keras
from keras.models import load_model
from keras.applications.vgg16 import VGG16, decode_predictions
# from Project.clu_util import cluster_util   # app.py와 __init__.py에서 다름
# from Project.mnist_util import mnist_util
import keras.backend.tensorflow_backend as tb
app = Flask(__name__)



@app.route('/classification_iris', methods=['GET', 'POST'])
def classification_iris():
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':True, 'clst':False, 'user':False}
    if request.method == 'GET':
        return render_template('classification_iris.html', menu=menu)
    else:
        sp_names = ['Setosa', 'Versicolor', 'Virginica']
        slen = float(request.form['slen'])      # Sepal Length
        swid = float(request.form['swid'])      # Sepal Width
        plen = float(request.form['plen'])      # Petal Length
        pwid = float(request.form['pwid'])      # Petal Width
        test_data = np.array([slen, swid, plen, pwid]).reshape(1,4)
        species_lr = sp_names[model_iris_lr.predict(test_data)[0]]
        species_svm = sp_names[model_iris_svm.predict(test_data)[0]]
        species_dt = sp_names[model_iris_dt.predict(test_data)[0]]
        species_deep = sp_names[model_iris_deep.predict_classes(test_data)[0]]
        iris = {'slen':slen, 'swid':swid, 'plen':plen, 'pwid':pwid, 
                'species_lr':species_lr, 'species_svm':species_svm,
                'species_dt':species_dt, 'species_deep':species_deep}
        return render_template('cla_iris_result.html', menu=menu, iris=iris)

# @app.route('/')
# def project():
    # return render_template('base.html')

# @app.route('/')
# def project():
    # return render_template('home.html')






'if __name__ == '__main__':
    load_movie_lr()
    load_movie_nb()
    load_iris()
    load_mnist()
    app.run()  