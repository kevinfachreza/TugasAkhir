"""
FIX MENGGUNAKAN BERNOULLI
HASIL LEBIH BAGUS
DENGAN PROBABILITY YANG LEBIH BAIK JUGA

"""

#python.exe "C:\Users\Kevin\PycharmProjects\TugasAkhir\NaiveBayes\index-test.py"

#Import Library of Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
import numpy
import pandas
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

t_series = ['t3','t5','t8','t10','t12','t15','t18','t20','t23']

for t in t_series:

    #assigning predictor and target variables
    dataframe = pandas.read_csv("../Users/Kevin/PycharmProjects/TugasAkhir/dataset/classifier-training-"+t+".csv", skipinitialspace=True)
    dataset = dataframe.values
    jumlah_gejala = len(dataset[0]) - 1

    X_train = dataset[:,0:jumlah_gejala]
    Y_train = dataset[:,jumlah_gejala]


    #assigning testing
    dataframe = pandas.read_csv("../Users/Kevin/PycharmProjects/TugasAkhir/dataset/classifier-testing-"+t+".csv", skipinitialspace=True)
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
    filewrite = '../Users/Kevin/PycharmProjects/TugasAkhir/laporan/'+t+'/nb_prediction_result_only.txt'
    with open(filewrite, 'w') as result_file:

        item_pred_IR = 0;
        pred_class_score_array = []
        item_count = 0;
        for pred in predictions:
            top5 = pred.argsort()[-5:][::-1]

            item_pred = top5[0]
            item_pred_str = class_map[item_pred]

            item_true = Y_test[index]

            string = str(item_pred_str)
            #print(string)
            result_file.write(string+'\n')

            pred_class_score = 100;
            pred_class_score_fix = 0;
            for item in top5:
                #print (class_map[item], pred[item])

                temp_pred = str(class_map[item])
                temp_true = str(item_true)
                
                if(pred_class_score_fix == 0):
                    if(temp_true == temp_pred):
                        pred_class_score_fix = pred_class_score
                        item_pred_IR += 1
                    else:
                        pred_class_score = pred_class_score - 20;




                #print (item)
            
            pred_class_score_array.append(pred_class_score_fix)
            #print ("")
            #print ("")

            index += 1


    print("PREDICT RESULT DONE")

filewrite = '../Users/Kevin/PycharmProjects/TugasAkhir/laporan/true/nb_prediction_result_only.txt'
with open(filewrite, 'w') as result_file:
    for y_true in Y_test:
        no_index = numpy.where(class_map == y_true)
        no_index = no_index[0][0]
        y = str(y_true)
        result_file.write(y+'\n')
