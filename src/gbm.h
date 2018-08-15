#ifndef GBM_H
#define GBM_H

#include <vector>
#include <string>
#include "tree.h"

using std::vector;
using std::string;

class GBM
{
public:
    GBM();
    GBM(int objective, double learning_rate, unsigned int max_depth, double lambda, double min_split_gain, unsigned int num_boost_round);
    void init(int objective, double learning_rate, unsigned int max_depth, double lambda, double min_split_gain, unsigned int num_boost_round);
    void fit(vector<vector<double>> &X, vector<double> &y);
    vector<double> predict(vector<vector<double>> &X);

private:
    vector<double> getScores(vector<vector<double>> &X, int iteration);
    vector<double> getGrad(vector<double> &y, vector<double> &scores);
    vector<double> getL2lossGrad(vector<double> &y, vector<double> &y_hat);
    vector<double> getLoglossGrad(vector<double> &y, vector<double> &y_hat);
    vector<double> getHessian(vector<double> &y, vector<double> &scores);
    vector<double> getL2lossHessian(vector<double> &y, vector<double> &y_hat);
    vector<double> getLoglossHessian(vector<double> &y, vector<double> &y_hat);
    double getMetric(vector<double> &y, vector<double> &y_hat);
    double  getError(vector<double> &y, vector<double> &y_hat);
    double getMSE(vector<double> &y, vector<double> &y_hat);
    
    int objective;
    double learning_rate;
    unsigned int  max_depth;
    double lambda;
    double min_split_gain;
    unsigned int num_boost_round;
    unsigned int best_iteration;
    
    Tree *trees;
};

#endif 
