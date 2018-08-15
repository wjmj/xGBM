#!/usr/bin/env python
from  xgbm import *
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


train = pd.read_csv('data/regression.train', header=None, sep='\t')
test = pd.read_csv('data/regression.test', header=None, sep='\t')
print(train.shape)
print(test.shape)

y_train = train[0].values
y_test = test[0].values
X_train = train.drop(0, axis=1).values
X_test = test.drop(0, axis=1)

model = xgbm(objective=1, learning_rate=0.3, max_depth=5, lambd=.1, min_split_gain=0.1, num_boost_round=20)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('error:', 1 - accuracy_score(y_test, y_pred))

