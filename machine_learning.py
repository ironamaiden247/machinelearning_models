import pandas as pd
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

from sklearn.neural_network import MLPClassifier

le = LabelEncoder()
## Get the data
app_data = pd.read_csv(r"C:\Saurabh\MLPriorityPre\VodafoneNZ_Internal_Pre.csv")
app_data = app_data.apply(le.fit_transform)

## Prepare input set X
X = app_data.drop(columns=['Priority'])
# print(X)
# ## Prepare the output set Y
Y = app_data['Priority']

# print(Y)
# ## Create the machine learning model
model = DecisionTreeClassifier()
model.fit(X,Y)

# model = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
# model.fit(X, Y)

## Provide the learning data to the model
# model.fit(X, Y)

# model = svm.SVC()
# model.fit(X,Y)

# model = RandomForestClassifier()
# model.fit(X, Y)

# model = DecisionTreeClassifier()
# model.fit(X, Y)

# model = LogisticRegression()
# model.fit(X,Y)

## Measure accuracy, precision, recall and f1 score of the model - Split dataset into 10 folds
kfold = KFold(n_splits=10, shuffle=True)

accuracy = cross_val_score(model, X, Y, cv=kfold)
print('Accuracy', np.mean(accuracy), accuracy)
recall = cross_val_score(model, X, Y, cv=kfold, scoring='recall_micro')
print('Recall', np.mean(recall), recall)
precision = cross_val_score(model, X, Y, cv=kfold, scoring='precision_micro')
print('Precision', np.mean(precision), precision)
f1 = cross_val_score(model, X, Y, cv=kfold, scoring='f1_micro')
print('F1', np.mean(f1), f1)

