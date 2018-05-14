#Import Library of Gaussian Naive Bayes model
from sklearn.naive_bayes import BernoulliNB
import numpy
import pandas
import pickle
from sklearn.model_selection import train_test_split

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#assigning predictor and target variables
dataframe = pandas.read_csv("../Users/Kevin/PycharmProjects/TugasAkhir/NN/dataset-23Apr.csv", skipinitialspace=True,
                             skiprows=1 )
dataset = dataframe.values
X = dataset[:,0:376]
Y = dataset[:,376]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)


#label
label = pandas.read_csv("../Users/Kevin/PycharmProjects/TugasAkhir/NN/label-23Apr.csv")
label = label.values

#Create a Gaussian Classifier
model = BernoulliNB()

# Train the model using the training sets 
score = model.fit(X_train, Y_train).score(X_test, Y_test)
print(score)

#---------------------------------------------------------------------------
#save model
#filename = '../Users/Kevin/PycharmProjects/TugasAkhir/NaiveBayes/model_architecture.sav'
#pickle.dump(model, open(filename, 'wb'))

# loading model
#model = pickle.load(open(filename, 'rb'))
#score = model.score(X_test, Y_test)

#print(score)

#---------------------------------------------------------------------------


"""
predictions = model.predict_proba(X_test)

#reverse encoding kalo mau diprint 1 1 tiap baris
for pred in predictions:
    	top5 = pred.argsort()[-5:][::-1]
    	print (top5)
    	for item in top5:
    		print (label[item], pred[item])
    		#print (item)
    	print ("")
    	print ("")

print(predictions)

"""