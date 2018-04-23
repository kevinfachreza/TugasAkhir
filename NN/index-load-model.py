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

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
dataframe = pandas.read_csv("../Users/Kevin/PycharmProjects/TugasAkhir/NN/datatest-23Apr.csv")
dataset = dataframe.values
X = dataset[:,0:376]
print(X)
Y = dataset[:,376]


label = pandas.read_csv("../Users/Kevin/PycharmProjects/TugasAkhir/NN/label-23Apr.csv")
label = label.values
#X = preprocessing.scale(X)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# convert integers to dummy variables (i.e. one hot encoded)
y = np_utils.to_categorical(encoded_Y)

def load_model():
    # loading model
    model = model_from_json(open('model_architecture.json').read())
    model.load_weights('model_weights.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model



model = load_model()

# predictions

#predictions = model.predict_classes(X, verbose=0)
#print(predictions)

predictions = model.predict(X, verbose=0)
#print(predictions)

#reverse encoding kalo mau diprint 1 1 tiap baris
for pred in predictions:
    	top5 = pred.argsort()[-5:][::-1]
    	for item in top5:
    		print (label[item], pred[item])
    	print ("")
    	print ("")

print(X[0])