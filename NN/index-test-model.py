import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
dataframe = pandas.read_csv("../Users/Kevin/PycharmProjects/TugasAkhir/NN/dataset-23Apr.csv", skipinitialspace=True,
                             skiprows=1 )
dataset = dataframe.values

jumlah_atribut = 376

X = dataset[:,0:jumlah_atribut]
Y = dataset[:,jumlah_atribut]

#data perlu di encode karena datanya berupa multiclass jadi harus pake categorical_crossentropy
#encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# define baseline model
def baseline_model():
	# create model
	model = Sequential()

	total_input = 376
	hidden_layer = 8
	class_output = 72

	model.add(Dense(hidden_layer, input_dim=total_input, activation='relu'))
	model.add(Dense(class_output, activation='softmax'))

	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	print ("done remodel")

	return model


print ("")
print ("====================================")
print ("START")
print ("====================================")
print ("")


#train model and create model as estimator
estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)

#estimator harus di fit agar bisa di save jadi model
estimator.fit(X,Y)

print ("")
print ("====================================")
print("TESTING")
print ("====================================")
print ("")

#testing pake kfold
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))



