#include <iostream>
#include <cmath>
#include "tree.h"
#include "utils.h"

using namespace std;

Tree::Tree()
{
    root = nullptr;
}

Tree::~Tree()
{
    if (root != nullptr)
        delete root;    
}


void Tree::fit(vector<vector<double>> &X, vector<double> &grad, vector<double> &hessian, unsigned int max_depth, double lambda, double min_split_gain)
{
    if (root != nullptr)
        delete root;

    root = new Node();
    this->max_depth = max_depth;
    this->lambda = lambda;
    this->min_split_gain = min_split_gain;
    fit(X, grad, hessian, root, 0);
}

void Tree::fit(vector<vector<double>> &X, vector<double> &grad, vector<double> &hessian, Node* node, unsigned int current_depth)
{
    double G = Utils::sum(grad);
    double H = Utils::sum(hessian);
    if (current_depth >=  max_depth || X.size() == 1)
    {
        node->isLeaf = true;
        node->weight = getLeafWeight(G, H, lambda);
        return; 
    }

    double best_gain = -1.;
    int best_feature_id = -1;
    double best_val = 0.;
    int num_samples = X.size();
    int num_features = X[0].size();
    vector<int> best_left_instance_ids;
    vector<int> best_right_instance_ids;
#pragma omp parallel for
    for (int feature_id = 0; feature_id < num_features; feature_id++)
    {
        double G_l = 0.;
        double H_l = 0.;
        vector<int> sorted_instance_ids = Utils::argsort(Utils::getColumn(X, feature_id));
        for (auto it = sorted_instance_ids.begin(); it != sorted_instance_ids.end(); it++)
        {
            int sample_id = *it;
            G_l += grad[sample_id];
            H_l += hessian[sample_id];
            double G_r = G - G_l;
            double H_r = H - H_l;
            double current_gain = getSplitGain(G, H, G_l, H_l, G_r, H_r, lambda);
#pragma omp critical
            {
            if (current_gain > best_gain)
            {
                best_gain = current_gain;
                best_feature_id = feature_id;
                best_val = X[sample_id][feature_id];
                best_left_instance_ids.assign(sorted_instance_ids.begin(), it+1);
                best_right_instance_ids.assign(it+1, sorted_instance_ids.end());
            }
            }

        }

    }

    if (best_gain < 0. || best_gain < min_split_gain || best_left_instance_ids.size()== 0 || best_right_instance_ids.size() == 0)
    {
        node->isLeaf = true;
        node->weight = getLeafWeight(G, H, lambda);
        return;
    }

    node->splitFeatureId = best_feature_id;
    node->splitVal = best_val;

    node->lchild = new Node();    
    vector<vector<double>> X_l = Utils::slice(X, best_left_instance_ids);
    vector<double> grad_l = Utils::slice(grad, best_left_instance_ids);
    vector<double> hessian_l = Utils::slice(hessian, best_left_instance_ids);
    fit(X_l, grad_l, hessian_l, node->lchild, current_depth+1);

    node->rchild = new Node();
    vector<vector<double>> X_r = Utils::slice(X, best_right_instance_ids);
    vector<double> grad_r = Utils::slice(grad, best_right_instance_ids);
    vector<double> hessian_r = Utils::slice(hessian, best_right_instance_ids);
    fit(X_r, grad_r, hessian_r, node->rchild, current_depth+1);
}

double Tree::predict(vector<double> &x)
{
    if (root == nullptr)
        return 0.;
    return predict(x, root);
}

double Tree::predict(vector<double> &x, Node* node)
{
    if (node->isLeaf)
        return node->weight;

    if (x[node->splitFeatureId] <= node->splitVal)
        return predict(x, node->lchild);

    else
        return predict(x, node->rchild);
}

double Tree::computeTerm(double g, double h, double lambda)
{
    return pow(g, 2) / (h + lambda);
}

double Tree::getSplitGain(double G, double H, double Gleft, double Hleft, double Gright, double Hright, double lambda)
{
    return computeTerm(Gleft, Hleft, lambda) + computeTerm(Gright, Hright, lambda) - computeTerm(G, H, lambda);
}

double Tree::getLeafWeight(double G, double H, double lambda)
{
    return -G / (H + lambda);
}
