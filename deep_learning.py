import pandas as pd

import csv

import numpy as np
from numpy import asarray
from numpy import zeros

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier

from simpletransformers.classification import ClassificationModel


#######Deep learning models - MLP and BERT  - Run only one model at a time, comment the rest #######


#### Multi-layer perceptron (MLP) algorithm that trains using Backpropagation

# Encoding for learning and prediction for some models
# le = LabelEncoder()
# ## Get the data
# app_data = pd.read_csv(r"C:\Saurabh\MLPriorityPre\VodafoneNZ_Internal_Pre.csv")
# app_data = app_data.apply(le.fit_transform)
# print(app_data)

## Prepare input set X
# X = app_data.drop(columns=['Priority'])
# # print(X)
# # ## Prepare the output set Y
# Y = app_data['Priority']
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
# clf.fit(X, Y)
# kfold = KFold(n_splits=10, shuffle=True)
# accuracy = cross_val_score(clf, X, Y, cv=kfold)
# print('Accuracy', np.mean(accuracy), accuracy)
# recall = cross_val_score(clf, X, Y, cv=kfold, scoring='recall_micro')
# print('Recall', np.mean(recall), recall)
# precision = cross_val_score(clf, X, Y, cv=kfold, scoring='precision_micro')
# print('Precision', np.mean(precision), precision)
# f1 = cross_val_score(clf, X, Y, cv=kfold, scoring='f1_micro')
# print('F1', np.mean(f1), f1)



###BERT Model (This model will take hours to run to give the output)
dataset = []

with open("C:\Saurabh\MLPriorityPre\VodafoneNZ_Internal_Pre.csv", 'r') as f:
    for entry in f:
        review_reader = csv.reader(f, delimiter=',', quotechar='"')
        next(review_reader, None)  # Skip csv headers
        for entry in review_reader:
            label = int(entry[0])
            review = entry[1]
            data = [review, label]
            dataset.append(data)

# print(dataset)
dataset = pd.DataFrame(dataset)
# print(dataset)
#Prepare metrics
accuracy = []
recall = []
precision = []
f_measure=[]
#
# #Bert model and crossvalidation
kfold = KFold(n_splits=10, shuffle=True)
for t_index, v_index in kfold.split(dataset):
  # splitting Dataframe (dataset)
    t_df = dataset.iloc[t_index]
    v_df = dataset.iloc[v_index]
    # Defining Model
    model = ClassificationModel('bert', 'bert-base-uncased', use_cuda=False, args={'fp16': False})
  # train the model
    model.train_model(dataset)
  # Get crossvalidation results
    accuracy = cross_val_score(model, t_df, v_df, cv=kfold)
    accuracy.append(accuracy)
    recall = cross_val_score(model, t_df, v_df, cv=kfold, scoring='recall_micro')
    recall.append(recall)
    precision = cross_val_score(model, t_df, v_df, cv=kfold, scoring='precision_micro')
    precision.append(precision)
    f_measure = cross_val_score(model, t_df, v_df, cv=kfold, scoring='f1_micro')
    f_measure.append(f_measure)

print('Accuracy', np.mean(accuracy), accuracy)
print('Recall', np.mean(recall), recall)
print('Precision', np.mean(precision), precision)
print('F1', np.mean(f_measure), f_measure)








