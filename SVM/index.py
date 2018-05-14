"""
FIX MENGGUNAKAN BERNOULLI
HASIL LEBIH BAGUS
DENGAN PROBABILITY YANG LEBIH BAIK JUGA

"""

#python.exe "C:\Users\Kevin\PycharmProjects\TugasAkhir\SVM\index.py"

#Import Library of Gaussian Naive Bayes model
from sklearn.svm import SVC
import numpy
import pandas
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#assigning predictor and target variables
dataframe = pandas.read_csv("../Users/Kevin/PycharmProjects/TugasAkhir/dataset/classifier-training-2.csv", skipinitialspace=True)
dataset = dataframe.values
X_train = dataset[:,0:388]
Y_train = dataset[:,388]

#assigning testing
dataframe = pandas.read_csv("../Users/Kevin/PycharmProjects/TugasAkhir/dataset/classifier-testing-2.csv", skipinitialspace=True)
dataset = dataframe.values
X_test = dataset[:,0:388]
Y_test = dataset[:,388]

#label
label = pandas.read_csv("../Users/Kevin/PycharmProjects/TugasAkhir/dataset/label.csv")
label = label.values

#Create a Gaussian Classifier
model = SVC(kernel = 'linear', probability=True).fit(X_train, Y_train)

# Train the model using the training sets 
y_pred = model.fit(X_train, Y_train).predict(X_test)
class_map = model.classes_
score = model.score(X_test, Y_test)

#---------------------------------------------------------------------------
#save model
filename = '../Users/Kevin/PycharmProjects/TugasAkhir/NaiveBayes/model_architecture.sav'
pickle.dump(model, open(filename, 'wb'))

# loading model
loaded_model = pickle.load(open(filename, 'rb'))
score2 = loaded_model.score(X_test, Y_test)

#print(score)

#---------------------------------------------------------------------------

predictions = model.predict_proba(X_test)


#print hasil
index = 0
for pred in predictions:
    top5 = pred.argsort()[-5:][::-1]

    item_pred = top5[0]
    item_pred_str = class_map[item_pred]

    item_true = Y_test[index]

    string = str(item_pred_str) + ' --- ' + str(item_true)
    print(string)

    for item in top5:
        print (class_map[item], pred[item])
        #print (item)
    print ("")
    print ("")

    index += 1

print(score)
print(score2)


#-----------------------------------------
#PRINT CONFUSION MATRIX
#-----------------------------------------

#init for predict, karena yang lama pake predict proba jadi beda
predict = model.predict(X_test)

#generate confusion
cm = confusion_matrix(Y_test, predict)

#format confusion jadi pandas supaya lebih rapi
y_true = pandas.Series(Y_test)
y_pred = pandas.Series(predict)

cm_pd = pandas.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)

#ngambil semua nama kolom karena urutan beda
table_rows = cm_pd.index.values

#init size
size = len(cm[0])

#mencari panjang N dari NxN matriks
total_val = 0
for i in range(size):
	for j in range(size):
		val = cm[i][j]
		total_val += val

#ambil data fp fn tn tp

for i in range(size):
	row = cm[i]
	TP = cm[i][i]

	#calculate FP
	FN = 0
	for j in range(size):
		val = cm[i][j]
		if(i != j):
			FN = FN + val

	FP = 0
	for j in range(size):
		val = cm[j][i]
		if(i != j):
			FP = FP + val

	TN = total_val - TP - FP - FN
	print(table_rows[i])
	print ("TP :",TP)
	print ("FP :",FP)
	print ("FN :",FN)
	print ("TN :",TN)
	print("")
	print("")


#printout semua confusion matrix
print (cm_pd)