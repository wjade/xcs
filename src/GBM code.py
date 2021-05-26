#build GBM model and manual tuning functions
import numpy as np
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost
from xgboost import XGBClassifier
import math
import collections
from collections import namedtuple

#executing data cleaning code
#exec(open("/Users/yejiang/Desktop/Stanford ML class/project/code/data cleaning.py").read())

#using default parameters as baseline model
model_baseline = XGBClassifier(base_score=0.5

                              , colsample_by_tree = 1
                              , subsample = 0.9
                              , gamma = 0
                              , learning_rate = 0.3
                              , max_depth = 6
                              , min_child_weight = 1
                              
                              , missing=None
                              , n_estimators=100
                              , nthread=-1
                              , max_delta_step=0
                              , objective='binary:logistic'
                              , reg_alpha=0
                              , reg_lambda=1
                              , scale_pos_weight=1
                              , seed=42
                              , silent=True)

model_baseline.fit(X_train, Y_train)

train_pred = model_baseline.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_pred)#0.9649
print(train_accuracy)

dev_pred = model_baseline.predict(X_dev)
dev_accuracy = accuracy_score(Y_dev, dev_pred)#0.9578
print(dev_accuracy)

test_pred = model_baseline.predict(X_test)
test_accuracy = accuracy_score(Y_test, test_pred)#0.9584
print(test_accuracy)


#grid search for parameter tuning

grid_search = []

i = 0

for col_sample in (0.7, 1):#baseline is 1
    for row_sample in (0.5, 0.9): #baseline is 0.9
        for g in (0, 5): #baseline is 0
            for eta in (0.05, 0.3): #baseline is 0.3
                for d in (4, 6): #baseline is 6
                    for c in (1, 5): #baseline is 1

                        i = i + 1
                        
                        model_grid = XGBClassifier(base_score=0.5

                              , colsample_by_tree = col_sample
                              , subsample = row_sample
                              , gamma = g
                              , learning_rate = eta
                              , max_depth = d
                              , min_child_weight = c
                              
                              , missing=None
                              , n_estimators=100
                              , nthread=-1
                              , max_delta_step=0
                              , objective='binary:logistic'
                              , reg_alpha=0
                              , reg_lambda=1
                              , scale_pos_weight=1
                              , seed=42
                              , silent=True)


                        model_grid.fit(X_train, Y_train)

                        train_pred = model_grid.predict(X_train)
                        train_accuracy = accuracy_score(Y_train, train_pred)

                        dev_pred = model_grid.predict(X_dev)
                        dev_accuracy = accuracy_score(Y_dev, dev_pred)

                        grid_search.append([col_sample, row_sample, g, eta, d, c, train_accuracy, dev_accuracy])

                        print(i)


grid_search = np.array(grid_search)
print(grid_search)

#returns grid_search with the max
max_value = grid_search[grid_search[:,7]==grid_search[:,7].max()]

#max value1
#[0.7, 0.9, 5, 0.3, 6, 5, 0.9622, 0.9585]
#[1,    0.9, 5, 0.3, 6, 5, 0.9622, 0.9585]

#grid search's result is slightly better than baseline, imporved from 0.9578 to 0.9585
#comparing the parameters, the baseline model appears to be overfitting(parameters are telling the same story)
