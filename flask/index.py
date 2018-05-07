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
	label = pandas.read_csv("../Users/Kevin/PycharmProjects/TugasAkhir/NN/label-23Apr.csv")
	label = label.values
	
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

	#print (gejala_array_np)

	#LOAD MODEL
	filename = '../Users/Kevin/PycharmProjects/TugasAkhir/NaiveBayes/model_architecture.sav'
	
	loaded_model = pickle.load(open(filename, 'rb'))

	predictions = loaded_model.predict_proba(gejala_array_np)

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

    			json_item_string = '{"diagnosis":"' + str(labels) +'","probability":"'+ str(pred[item]) +'"}'
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