# MLP for Pima Indians Dataset Serialize to JSON and HDF5
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import numpy
import os
import pandas

print ("")
print ("====================================")
print ("START")
print ("====================================")
print ("")


# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataframe = pandas.read_csv("../Users/Kevin/PycharmProjects/AqeelaTugasAkhir/NN/dataset-3Apr.csv", skipinitialspace=True,
                             skiprows=1 )
dataset = dataframe.values

# split into input (X) and output (Y) variables
X = dataset[:,0:229].astype(int)
Y = dataset[:,229]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# create model
model = Sequential()
## Layer 1
### Neuro 458, dengan input 229

model.add(Dense(8, input_dim=229, kernel_initializer='uniform', activation='relu'))
model.add(Dense(28, kernel_initializer='uniform', activation='softmax'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))



# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
#iterasi 150x dengan batck 10
model.fit(X, Y, epochs=200, batch_size=5, verbose=0)

# evaluate the model
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

 
# serialize model to JSON
model_json = model.to_json()
with open("../Users/Kevin/PycharmProjects/AqeelaTugasAkhir/NN/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("../Users/Kevin/PycharmProjects/AqeelaTugasAkhir/NN/model.h5")
print("Saved model to disk")
 
# later...
 
# load json and create model
json_file = open('../Users/Kevin/PycharmProjects/AqeelaTugasAkhir/NN/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("../Users/Kevin/PycharmProjects/AqeelaTugasAkhir/NN/model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))