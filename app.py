from flask import Flask, render_template, redirect, request
from keras.layers import Dense, Input, Conv2D, LSTM, MaxPool2D, UpSampling2D
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from numpy import argmax, array_equal
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import Model
from imgaug import augmenters
import random
from datetime import datetime
from numpy import genfromtxt
import pandas as pd
import numpy as np

app = Flask(__name__)

my_data = genfromtxt('validation_MINIST.csv', delimiter=',')
model = load_model('model.h5')
preds = model.predict(my_data)

global_paths = []

@app.route('/', methods = ['GET', 'POST'])
def index():
	#random.seed(datetime.now())
	get_random = random.randint(0, 600)
	# print(get_random)
	f, ax = plt.subplots(1)
	ax.imshow(my_data[get_random].reshape(28, 28))
	path1 = "./static/{}q.png".format(get_random)
	path2 = "./static/{}a.png".format(get_random)
	plt.savefig(path1 , bbox_inches='tight')

	f, ax = plt.subplots(1)
	ax.imshow(preds[get_random].reshape(28, 28))
	plt.savefig(path2 , bbox_inches='tight')

	paths = [];

	paths.append(path1)
	paths.append(path2)

	global_paths.append(path1)
	global_paths.append(path2)
	print(global_paths)
	return render_template("index.html", paths = paths)


@app.route('/gallery', methods = ['GET', 'POST'])
def gallery():
    return render_template("gallery.html", paths = global_paths)
# @app.route('/', methods = ['POST'])
# def upload():
# 	if request.method == 'POST':

# 		f = request.files['userfile']
# 		path = "./static/{}".format(f.filename)
# 		f.save(path)
# 		#paths = []
# 	return render_template("index.html", imgpath = path)

if __name__=='__main__':
	app.run(debug=True)













