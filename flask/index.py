from flask import Flask
from flask import request
from flask import jsonify
import json
# Train model and make predictions
import numpy
import pandas
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.utils import np_utils
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/predict", methods=['POST'])
def predict():
	#INIT
	total_attributes=376
	
	#RECEIVE DATA FROM REQUEST
	data = request.json
	data_gejala = data.get("gejala")

	#PROCESS DATA INTO ARRAY BIAR SESUAI FORMAT DARI PERMINTAAN PREDICT
	gejala_array = []
	for i in range(total_attributes):	
		number = str(i)
		if(data_gejala.get(number)):
			value_gejala = data_gejala.get(number)
		else:
			value_gejala = 0

		gejala_array.append(value_gejala)


	gejala_array_np_dummy = []
	gejala_array_np_dummy.append(gejala_array)

	gejala_array_np = numpy.array(gejala_array_np_dummy)


	#PREPROCESS DATA
	#gejala_array_np = preprocessing.scale(gejala_array_np)

	print (gejala_array_np)

	#LOAD MODEL
	model = model_from_json(open('model_architecture.json').read())
	model.load_weights('model_weights.h5')
	model.compile(loss='categorical_crossentropy', optimizer='adam')

	predictions = model.predict(gejala_array_np, verbose=0)
	#print(predictions)

	#reverse encoding kalo mau diprint 1 1 tiap baris
	print ("")
	print ("")
	for pred in predictions:
		top5 = pred.argsort()[-5:][::-1]
		for item in top5:
			print (item, pred[item])

	print ("")
	print ("")

	return jsonify(gejala_array)

if __name__ == "__main__":
    app.run()