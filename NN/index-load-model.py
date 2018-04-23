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

dataframe = pandas.read_csv("../Users/Kevin/PycharmProjects/TugasAkhir/NN/dataset-23Apr.csv", skipinitialspace=True,
                             skiprows=1 )
dataset = dataframe.values
X = dataset[:,0:376]
Y = dataset[:,376]

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


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=seed)

model = load_model()

# predictions

predictions = model.predict_classes(X_test, verbose=0)
print(predictions)

predictions = model.predict(X_test, verbose=0)
print(predictions)

#reverse encoding kalo mau diprint 1 1 tiap baris
for pred in predictions:
    	top5 = pred.argsort()[-5:][::-1]
    	for item in top5:
    		print (item, pred[item])
    	print ("")
    	print ("")

print(X_test[0])