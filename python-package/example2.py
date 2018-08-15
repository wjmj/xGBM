#!/usr/bin/env python
from xgbm import *
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
import time

X, y = load_boston(return_X_y=True)
print(X.shape)
print(y.shape)

model = xgbm(objective=0, learning_rate=0.1, max_depth=5, lambd=1., min_split_gain=.1, num_boost_round=30)
s = time.time()
model.fit(X, y)
print(time.time() - s)
