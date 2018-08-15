#include "gbm.h"

extern "C"
{
    GBM m;
    
    void fit(double** features, double* labels, int row, int col, int objective, double learning_rate, unsigned int max_depth, double lambda, double min_split_gain, unsigned int num_boost_round)
    {
        m.init(objective, learning_rate, max_depth, lambda, min_split_gain, num_boost_round);
        vector<vector<double>> X(row, vector<double>(col));
        for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            X[i][j] = features[i][j];

        vector<double> y(row);
        for (int i = 0; i < row; i++)
            y[i] = labels[i];

        m.fit(X, y);
    }

    void  predict(double **features, int row, int col, double *ret)
    {
        vector<vector<double>> X(row, vector<double>(col));
        for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            X[i][j] = features[i][j];
        vector<double> pred = m.predict(X);
        for (int i = 0; i < row; i++)
            ret[i] = pred[i];
    }
}
