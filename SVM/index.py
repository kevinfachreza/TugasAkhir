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
dataframe = pandas.read_csv("../Users/Kevin/PycharmProjects/TugasAkhir/dataset/classifier-training-t18.csv", skipinitialspace=True)
dataset = dataframe.values
jumlah_gejala = len(dataset[0]) - 1

X_train = dataset[:,0:jumlah_gejala]
Y_train = dataset[:,jumlah_gejala]

#assigning testing
dataframe = pandas.read_csv("../Users/Kevin/PycharmProjects/TugasAkhir/dataset/classifier-testing-t18.csv", skipinitialspace=True)
dataset = dataframe.values
X_test = dataset[:,0:jumlah_gejala]
Y_test = dataset[:,jumlah_gejala]

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
filewrite = '../Users/Kevin/PycharmProjects/TugasAkhir/laporan/t18/svm_prediction_result.txt'
with open(filewrite, 'w') as result_file:
    result_file.write('jumlah gejala ' + str(jumlah_gejala) + '\n')
    result_file.write('jumlah diagnosis ' + str(len(class_map)) + '\n')
    result_file.write('akurasi ' + str(score) + '\n\n')

    item_pred_IR = 0;
    pred_class_score_array = []
    item_count = 0;
    for pred in predictions:
        top5 = pred.argsort()[-5:][::-1]

        item_pred = top5[0]
        item_pred_str = class_map[item_pred]

        item_true = Y_test[index]

        string = str(item_pred_str) + ' --- ' + str(item_true)
        #print(string)
        result_file.write(string+'\n')

        pred_class_score = 100;
        pred_class_score_fix = 0;
        for item in top5:
            #print (class_map[item], pred[item])
            result_file.write(str(class_map[item]) + ' '+ str(pred[item])+'\n')
            temp_pred = str(class_map[item])
            temp_true = str(item_true)
            
            if(pred_class_score_fix == 0):
                if(temp_true == temp_pred):
                    pred_class_score_fix = pred_class_score
                    item_pred_IR += 1
                else:
                    pred_class_score = pred_class_score - 20;

        pred_class_score_array.append(pred_class_score_fix)
        result_file.write('skor IR' + str(pred_class_score_fix) + '\n')
        result_file.write('\n')
        #print ("")
        #print ("")

        index += 1

    pred_class_score_array_np = numpy.array(pred_class_score_array)
    result_file.write('item found ' + str(item_pred_IR) + '/'+ str(index) + '\n')
    result_file.write('item found skor ' + str(numpy.mean(pred_class_score_array_np)) + '\n')

print("PREDICT RESULT DONE")

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

filewrite = '../Users/Kevin/PycharmProjects/TugasAkhir/laporan/t18/svm_cm_class.txt'
with open(filewrite, 'w') as the_file:
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
        """
        print(table_rows[i])
        print ("TP :",TP)
        print ("FP :",FP)
        print ("FN :",FN)
        print ("TN :",TN)
        print("")
        print("")
        """

        the_file.write(table_rows[i]+'\n')
        the_file.write("TP :"+str(TP)+'\n')
        the_file.write("FP :"+str(FP)+'\n')
        the_file.write("FN :"+str(FN)+'\n')
        the_file.write("TN :"+str(TN)+'\n')
        the_file.write('\n')


print("CONFUSION MATRIX EACH CLASS DONE")


#printout semua confusion matrix dengan model pandas
#print (cm_pd)

#print confusion matrix


filewrite = '../Users/Kevin/PycharmProjects/TugasAkhir/laporan/t18/svm_cm_all.csv'
with open(filewrite, 'w') as new_file:
    #print("," + ",".join(class_map))
    new_file.write("," + ",".join(class_map) + '\n')
    for i in range(size):
        #print ( class_map[i] + ',' + ','.join(map(str, cm[i])))
        new_file.write ( class_map[i] + ',' + ','.join(map(str, cm[i]))  + '\n')


print("CONFUSION MATRIX DONE")