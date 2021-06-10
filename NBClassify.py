import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold, cross_val_score
import numpy as np

data = pd.read_csv(r"C:\Saurabh\MLPriorityPre\VodafoneNZ_External_Pre.csv")

data = data[['Priority','Reviews']]

cv=CountVectorizer()


X = cv.fit_transform(data['Reviews'])
Y = data['Priority']

model = MultinomialNB()
model.fit(X,Y)

kfold = KFold(n_splits=10, shuffle=True)


accuracy = cross_val_score(model, X, Y, cv=kfold)
print('Accuracy', np.mean(accuracy), accuracy)
recall = cross_val_score(model, X, Y, cv=kfold, scoring='recall_micro')
print('Recall', np.mean(recall), recall)
precision = cross_val_score(model, X, Y, cv=kfold, scoring='precision_micro')
print('Precision', np.mean(precision), precision)
f1 = cross_val_score(model, X, Y, cv=kfold, scoring='f1_micro')
print('F1', np.mean(f1), f1)