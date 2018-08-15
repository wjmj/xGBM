from ctypes import *
import os
import numpy as np

lib = cdll.LoadLibrary(os.path.dirname(os.path.realpath(__file__)) + '/lib_xgbm.so')

class xgbm(object):
    def __init__(self, objective=0, learning_rate=0.1, max_depth=5, lambd=1., min_split_gain=0.1, num_boost_round=30):
        self.objective = objective
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.lambd = lambd
        self.min_split_gain = min_split_gain
        self.num_boost_round = num_boost_round

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.double)
        y = np.asarray(y, dtype=np.double)

        row,col = X.shape
        double_p = cast(y.ctypes.data, POINTER(c_double))
        double_p_p = (X.ctypes.data + np.arange(X.shape[0]) * X.strides[0]).astype(np.uintp)

        DOUBLEPP = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags='C')
        lib.fit.argtypes = [DOUBLEPP, POINTER(c_double), c_int, c_int, c_int, c_double, c_uint, c_double, c_double, c_uint]
        lib.fit.restype = None
        task = 0
        if self.objective != 0:
            task = 1
        lib.fit(double_p_p, double_p, c_int(row), c_int(col), c_int(task), c_double(self.learning_rate), c_uint(self.max_depth),  
            c_double(self.lambd), c_double(self.min_split_gain), c_uint(self.num_boost_round))

    def predict(self, X):
        X = np.asarray(X, np.double)

        row, col = X.shape
        double_p_p = (X.ctypes.data + np.arange(X.shape[0]) * X.strides[0]).astype(np.uintp)
        ret = (c_double*row) (*([-1.0 for _ in range(row)]))
        ret_double_p = cast(ret, POINTER(c_double))

        DOUBLEPP = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags='C')
        lib.predict.argtypes = [DOUBLEPP, c_int, c_int, POINTER(c_double)]
        lib.predict.restype = None

        lib.predict(double_p_p, c_int(row), c_int(col), ret_double_p)

        return [ret_double_p[i] for i in range(row)]

