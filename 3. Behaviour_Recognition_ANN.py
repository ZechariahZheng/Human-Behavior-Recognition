import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.cross_validation import cross_val_predict

import matplotlib.pyplot as plt
import seaborn as sns

seed = 7
np.random.seed(seed)
total_components = 27
used_components = 27
file_path_train = "your_train_data.csv"
file_path_test = "your_test_data.csv"

dataframe_train = pd.read_csv(file_path_train, header=None)
dataset_train = dataframe_train.values
X_train = dataset_train[:, :used_components].astype(float)
X_train.shape
y_train = dataset_train[:, total_components + 1]
y_train.shape

dataframe_test = pd.read_csv(file_path_test, header=None)
dataset_test = dataframe_test.values
X_test = dataset_test[:, :used_components].astype(float)
X_test.shape
y_test = dataset_test[:, total_components + 1]
y_test.shape


# define baseline model


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(15, input_dim=used_components, activation='relu'))
    model.add(Dense(8, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=150, batch_size=15, verbose=0)

# test on my data
mean_test_acc = []
mean_confusion = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
whole_prediction = []
for i in range(10):
    estimator.fit(X_train, y_train, epochs=150, batch_size=15)
    predictions_test = estimator.predict(X_test)
    if i == 0:
        whole_prediction = predictions_test
    else:
        whole_prediction = np.row_stack((whole_prediction, predictions_test))
    print(accuracy_score(y_test, predictions_test))
    print(confusion_matrix(y_test, predictions_test))
    print(classification_report(y_test, predictions_test))
    mean_test_acc.append(accuracy_score(y_test, predictions_test))

mean_test_acc
np.mean(mean_test_acc)

whole_prediction

estimator.fit(X_train, y_train, epochs=200, batch_size=15)
predictions_train = estimator.predict(X_train)
print(accuracy_score(y_train, predictions_train))
print(confusion_matrix(y_train, predictions_train))
print(classification_report(y_train, predictions_train))

# Using Cross fold predict
y_val_pred = cross_val_predict(estimator, X_train, y_train, cv=10)
print(confusion_matrix(y_train, y_val_pred))﻿
print(accuracy_score(y_train, y_val_pred))
print(classification_report(y_train, y_val_pred))

y_score = cross_val_score(estimator, X_train, y_train, cv=10)
print y_score
print y_score.mean()


# Using Grid Search CV
param_grid = dict(epochs=[10, 20, 30], batch_size=[10, 20, 30])
grid = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)
grid_result.best_score_
grid_result.best_params_
grid_result.grid_scores_
