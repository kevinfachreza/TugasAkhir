# importing necessary libraries
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy
import pandas
 
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#assigning predictor and target variables
dataframe = pandas.read_csv("../Users/Kevin/PycharmProjects/TugasAkhir/NN/dataset-23Apr.csv", skipinitialspace=True,
                             skiprows=1 )
dataset = dataframe.values
X = dataset[:,0:376]
Y = dataset[:,376]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=seed)

 
# training a linear SVM classifier
from sklearn.svm import SVC
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, Y_train)
svm_predictions = svm_model_linear.predict(X_test)
 
# model accuracy for X_test  
accuracy = svm_model_linear.score(X_test, Y_test)
 
# creating a confusion matrix
cm = confusion_matrix(Y_test, svm_predictions)

print(accuracy)