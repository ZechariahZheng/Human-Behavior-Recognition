import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.font_manager
import seaborn as sns

import numpy as np

from sklearn import svm
from sklearn.grid_search import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import cross_val_predict

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

import time

num_train_data = 9999
num_test_data = 9999
num_beh_1_test_data = 9999
num_beh_2_test_data = 9999

# Fetch train and test data
file_path_train = "your_train_data.csv"
file_path_test = "your_test_data.csv"


components = 11
train_index = [4, 47, 48, 49, 50, 51, 52, 53]
test_index = [i for i in train_index]
test_index.append(54)
columns = range(44, 54)

X_train = pd.read_csv(file_path_train, header=None)[train_index]
X_train.shape
X_test = pd.read_csv(file_path_test, header=None)[test_index]
X_test.rename(columns={54: 'class'}, inplace=True)
X_test.shape

X_test_beh_1 = X_test.loc[X_test['class'] == 5]
X_test_beh_2 = X_test.loc[X_test['class'] == 6]



clf = svm.OneClassSVM(nu=0.063, kernel="rbf", gamma=0.000051)
clf.fit(X_train)

y_pred_train = clf.predict(X_train)
y_pred_test_all = clf.predict(X_test[train_index])

y_pred_test_beh_1 = clf.predict(X_test_beh_1[train_index])
y_pred_test_beh_2 = clf.predict(X_test_beh_2[train_index])



predict_train_rate = y_pred_train[y_pred_train == 1].size / num_train_data
predict_test_rate = y_pred_test_all[y_pred_test_all == 1].size / num_test_data

y_pred_test_beh_1[y_pred_test_beh_1 == 1].size / num_beh_1_test_data
y_pred_test_beh_2[y_pred_test_beh_2 == 1].size / num_beh_2_test_data


# pair plot
X_train['class'] = 1
X_test['class'] = -x1
pp_data = pd.concat([X_train, X_test])
sns.set(style='whitegrid', context='notebook')
sns.pairplot(pp_data, hue="class", size=4)
plt.tight_layout()
plt.show()

# Cross fold validation
val_X = pp_data.iloc[:, :components].values
val_y = pp_data.iloc[:, components].values

scores = cross_val_score(clf, val_X, val_y, cv=10, scoring='accuracy')
scores
scores.mean()
y_val_pred = cross_val_predict(clf, val_X, val_y, cv=10)
print "roc_auc:", roc_auc_score(val_y, y_val_pred)
print "recall_score:", recall_score(val_y, y_val_pred)
print(confusion_matrix(val_y, y_val_pred))﻿
print(accuracy_score(val_y, y_val_pred))
print(classification_report(val_y, y_val_pred))


np.linspace(0.001, 0.2, 100)
# Finding best nu and gamma
param_range = np.linspace(0.001, 0.2, 100)
param_range
search_X = pp_data.iloc[:, :components].values
search_y = pp_data.iloc[:, components].values
params = {'nu': param_range, 'gamma': param_range}
# GridSearchCV
grid = GridSearchCV(svm.OneClassSVM(), param_grid=params, scoring='recall')

start = time.clock()
grid.fit(search_X, search_y)
elapsed = (time.clock() - start)
print elapsed

grid.best_score_
grid.best_params_
grid.grid_scores_

# RandomizedSearchCV
rs = RandomizedSearchCV(svm.OneClassSVM(), param_distributions=params,
                        scoring='accuracy', n_iter=1000)
start = time.clock()
rs.fit(search_X, search_y)
elapsed = (time.clock() - start)
print elapsed

rs.grid_scores_
rs.best_score_
rs.best_params_



# Drawing 2D train graph
xx, yy = np.meshgrid(np.linspace(-6, 13, 500), np.linspace(-6, 7, 500))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("Non Target Detection")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

s = 40
b = plt.scatter(X_train.values[:, 0], X_train.values[:, 1], c='black')
c = plt.scatter(X_test.values[:, 0], X_test.values[:, 1], c='gold',)

plt.axis('tight')
plt.legend([a.collections[0], b,  c],
           ["learned frontier", "training observations",
            "new abnormal observations"],
           loc="upper left",
           prop=matplotlib.font_manager.FontProperties(size=11))
plt.xlabel(
    "Correct train target prediction: %d/" + num_train_data + "\n"
    "Correct test non target prediction: %d/" + num_test_data
    % (predict_train_rate, predict_test_rate))
plt.show()
