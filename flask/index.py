"""
KALO NGIRIM GEJALA
MISAL ID GEJALA NYA 3 DIKIRIM 2
GEJALA_ID 50 DIKIRIM 49
INTINYA DI KURANGI 1

"""

#python.exe "C:\Users\Kevin\PycharmProjects\TugasAkhir\flask\index.py"

from flask import Flask
from flask import request
from flask import jsonify
import json
import jsonpickle
from json import JSONEncoder
# Train model and make predictions
import numpy
import pandas
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.naive_bayes import BernoulliNB

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class HasilDiagnosis(object):
    def __init__(self, diagnosis, probability):
        self.diagnosis = diagnosis
        self.probability = probability

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/predict", methods=['POST'])
def predict():
	#INIT
	total_attributes=376

	#init labels
	label = pandas.read_csv("../Users/Kevin/PycharmProjects/TugasAkhir/dataset/label.csv")
	label = label.values

	#init attributes
	attributes = pandas.read_csv("../Users/Kevin/PycharmProjects/TugasAkhir/dataset/attributes-t5.csv")
	attributes = attributes.values

	
	#RECEIVE DATA FROM REQUEST
	data = request.json
	data_gejala = data.get("gejala")

	#PROCESS DATA INTO ARRAY BIAR SESUAI FORMAT DARI PERMINTAAN PREDICT
	number = str(5)
	gejala_array = []
	for i in attributes:	
		number = str(i[0])
		if(data_gejala.get(str(number))):
			value_gejala = data_gejala.get(number)
		else:
			value_gejala = 0

		gejala_array.append(value_gejala)


	gejala_array_np_dummy = []
	gejala_array_np_dummy.append(gejala_array)

	gejala_array_np = numpy.array(gejala_array_np_dummy)


	#PREPROCESS DATA
	#gejala_array_np = preprocessing.scale(gejala_array_np)

	#print (gejala_array_np)

	#LOAD MODEL
	# fix random seed for reproducibility
	seed = 7
	numpy.random.seed(seed)

	#assigning predictor and target variables
	dataframe = pandas.read_csv("../Users/Kevin/PycharmProjects/TugasAkhir/dataset/classifier-training-t5.csv", skipinitialspace=True)
	dataset = dataframe.values
	jumlah_gejala = len(dataset[0]) - 1

	X_train = dataset[:,0:jumlah_gejala]
	Y_train = dataset[:,jumlah_gejala]

	#assigning testing
	dataframe = pandas.read_csv("../Users/Kevin/PycharmProjects/TugasAkhir/dataset/classifier-testing-t5.csv", skipinitialspace=True)
	dataset = dataframe.values
	X_test = dataset[:,0:jumlah_gejala]
	Y_test = dataset[:,jumlah_gejala]

	#label
	label = pandas.read_csv("../Users/Kevin/PycharmProjects/TugasAkhir/dataset/label.csv")
	label = label.values

	#Create a Gaussian Classifier
	model = BernoulliNB()

	# Train the model using the training sets 
	y_pred = model.fit(X_train, Y_train).predict(X_test)
	class_map = model.classes_
	score = model.score(X_test, Y_test)

	predictions = model.predict_proba(gejala_array_np)

	#reverse encoding kalo mau diprint 1 1 tiap baris
	#karena bingung gimana format list ke json, jadi list dibiarin aja buat debugging, yang json di cetak secara string
	#print ("")
	#print ("")

	result = []
	jsondata = '{ "result":['

	index = 0
	for pred in predictions:
		top5 = pred.argsort()[-5:][::-1]
		for item in top5:
    			labels = label[item][0]
    			result.append(HasilDiagnosis(labels,pred[item]))

    			json_item_string = '{"diagnosis":"' + str(class_map[item]) +'","probability":"'+ str(pred[item]) +'"}'
    			if index < 4:
    				json_item_string = json_item_string + ','

    			#print (json_item_string)
    			index = index+1
    			jsondata = jsondata + json_item_string


	jsondata = jsondata + "]}" 
	#print ("")
	#print ("")
	#for i in range(5):
		#print (result[i].diagnosis, result[i].probability)

	#print (jsondata)

	format_jsondata = json.dumps(jsondata)

	return jsondata
	
if __name__ == "__main__":
    app.run()