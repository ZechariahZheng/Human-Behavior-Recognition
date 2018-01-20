import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm
from sklearn.grid_search import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import cross_val_predict
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import time

num_train_data = 9999
num_test_data = 9999

posture_file_path_train = "your_train_data.csv"
posture_file_path_test = "your_test_data.csv"
train_features = 54
test_features = 54
posture_train_data = pd.read_csv(posture_file_path_train, header=None)[range(0, train_features)]
posture_test_data = pd.read_csv(posture_file_path_test, header=None)[range(0, test_features)]


posture_train_data['class'] = 1
posture_test_data['class'] = -1
whole_data = pd.concat([posture_train_data, posture_test_data])

# Perform feature selection
selector = SelectKBest(f_classif, k=20)
new_data = selector.fit_transform(whole_data[range(train_features)], whole_data['class'])

len(new_data[:num_train_data, :])
new_X_train = new_data[:num_train_data, :]
new_X_train.shape
new_X_test = new_data[num_train_data:, :]
new_X_test.shape

clf = svm.OneClassSVM(nu=0.08, kernel="rbf", gamma=0.05)
clf.fit(new_X_train)

y_pred_train = clf.predict(new_X_train)
y_pred_test = clf.predict(new_X_test)

predict_train_rate = y_pred_train[y_pred_train == 1].size
predict_test_rate = y_pred_test[y_pred_test == -1].size

y_pred_train[y_pred_train == 1].size / num_train_data
y_pred_test[y_pred_test == -1].size / num_test_data


pp_X = np.vstack([new_X_train, new_X_test])
pp_X.shape
pp_y = [1 if x < num_train_data else -1 for x in range(num_train_data + num_test_data)]

scores = cross_val_score(clf, pp_X, pp_y, cv=10, scoring='f1_macro')
scores
scores.mean()
y_val_pred = cross_val_predict(clf, pp_X, pp_y, cv=10)
print roc_auc_score(pp_X, pp_y)
print(confusion_matrix(pp_X, pp_y))﻿
print(accuracy_score(pp_X, pp_y))
print(classification_report(pp_X, pp_y))
