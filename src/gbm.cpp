#include <iostream>
#include <ctime>
#include <random>
#include <limits>

#include "gbm.h"
#include "utils.h"

using namespace std;

GBM::GBM()
{
    GBM(0, 0.3, 5, 1., 0.1, 30);
}

GBM::GBM(int objective, double learning_rate, unsigned int max_depth, double lambda, double min_split_gain, unsigned int num_boost_round)
{
    this->objective = objective;
    this->learning_rate = learning_rate;
    this->max_depth = max_depth;
    this->lambda = lambda;
    this->min_split_gain = min_split_gain;
    this->num_boost_round = num_boost_round;
    trees = nullptr;
}

void GBM::init(int objective, double learning_rate, unsigned int max_depth, double lambda, double min_split_gain, unsigned int num_boost_round)
{
    this->objective = objective;
    this->learning_rate = learning_rate;
    this->max_depth = max_depth;
    this->lambda = lambda;
    this->min_split_gain = min_split_gain;
    this->num_boost_round = num_boost_round;
    trees = nullptr;
}

void GBM::fit(vector<vector<double>> &X, vector<double> &y)
{
    if (trees != nullptr)
        delete [] trees;

    trees = new Tree[num_boost_round];
    best_iteration = -1;
    double best_loss = numeric_limits<double>::max();
    cout << "Training" << endl;
    vector<double> y_pred;
    for (int iter = 0; iter < num_boost_round; iter++)
    {
        vector<double> grads = getGrad(y, y_pred);
        vector<double> hessian = getHessian(y, y_pred);
        trees[iter].fit(X, grads, hessian, max_depth, lambda, min_split_gain);
        y_pred = getScores(X, iter);
        
        double loss = getMetric(y, y_pred);
        string metric("MSE");
        if (objective != 0)
            metric = "error";
            
        cout << "Iteration: " << iter << ", " << metric << ": " << loss << endl;

        if (loss < best_loss)
        {
            best_loss = loss;
            best_iteration = iter;
        }

    }

    cout << "Training Finished" << endl;
}

vector<double> GBM::predict(vector<vector<double>>& X)
{
    int num_samples = X.size();
    vector<double> result(num_samples, 0.);
    for (int num = 0; num < num_samples; num++)
#pragma omp parallel for
    for (int i = 0; i < best_iteration; i++)
    {
        result[num] += trees[i].predict(X[num]) * learning_rate;
    }

    return result;
}

vector<double> GBM::getScores(vector<vector<double>>& X, int iter)
{
    vector<double> result;
    if (iter < 0)
        return result;
    
    int num_samples = X.size();
    vector<double> res(num_samples, 0.);
    for (int num = 0; num < num_samples; num++)
#pragma omp parallel for
    for (int i = 0; i <= iter; i++)
    {
        res[num] += trees[i].predict(X[num]) * learning_rate;
    }

    return res;
}

vector<double> GBM::getGrad(vector<double> &y, vector<double> &y_hat)
{
    if (y_hat.size() == 0)
    {
        unsigned int seed = time(NULL);
        int len = y.size();
        return Utils::random(len);
    }
    if (objective == 0)
        return getL2lossGrad(y, y_hat);
    return getLoglossGrad(y, y_hat);
}

vector<double> GBM::getL2lossGrad(vector<double>& y, vector<double>& scores)
{
    vector<double> e = Utils::subtract(scores, y);
    vector<double> grad = Utils::multiply(e, 2); 

    return grad;
}

vector<double> GBM::getLoglossGrad(vector<double> &y, vector<double> &y_hat)
{
    return Utils::subtract(y_hat, y);
}   

vector<double> GBM::getHessian(vector<double>& y, vector<double>& y_hat)
{
    if (objective == 0)
        return getL2lossHessian(y, y_hat);

    return getLoglossHessian(y, y_hat);
}

vector<double> GBM::getLoglossHessian(vector<double> &y, vector<double> &y_hat)
{
    if (y_hat.size() == 0)
    {
        unsigned int seed = time(NULL);
        int len = y.size();
        return Utils::random(len);
    }
    vector<double> h(y.size(), 0.);
    for (int i = 0; i < y.size(); i++)
    {
        h[i] = y_hat[i] * (1 - y_hat[i]);
    }

    return h;
}

vector<double> GBM::getL2lossHessian(vector<double> &y, vector<double> &y_hat)
{
    vector<double> h(y.size(), 2);

    return h;
}

double GBM::getMetric(vector<double> &y, vector<double> &y_hat)
{
    if (objective == 0)
        return getMSE(y, y_hat);
    return getError(y, y_hat);
}

double GBM::getError(vector<double> &y, vector<double> &y_hat)
{
    vector<double> y_pred(y_hat);
    for (int i = 0; i < y_pred.size(); ++i)
    {
        if (y_pred[i] > 0.5)
            y_pred[i] = 1;
        else
            y_pred[i] = 0;
    }
    int wrongCases = 0;
    for (int i = 0; i < y_pred.size(); ++i)
    {
        if (!Utils::same(y[i], y_pred[i]))
            wrongCases++;
    }

    return wrongCases * 1. / y.size();
}

double GBM::getMSE(vector<double>& y, vector<double>& y_hat)
{
    vector<double> e = Utils::subtract(y, y_hat);
    vector<double> e2 = Utils::square(e);
    return Utils::mean(e2);
}
