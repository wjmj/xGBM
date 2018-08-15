#ifndef TREE_H
#define TREE_H

#include <vector>

using std::vector;

struct Node
{
    bool isLeaf;
    Node *lchild;
    Node *rchild;
    int splitFeatureId;
    int splitVal;
    int weight;

    Node()
    {
        isLeaf = false;
        lchild = nullptr;
        rchild = nullptr;
        splitFeatureId = -1;
        splitVal = 0;
        weight = 0;    
    }

    ~Node()
    {
        if (lchild != nullptr)
            delete lchild;
        if (rchild != nullptr)
            delete rchild;
    }
};
 
class Tree
{
public:
    Tree();
    ~Tree();
    void fit(vector<vector<double>> &X, vector<double> &grad, vector<double> &hessian, unsigned int max_depth, double lambda, double min_split_gain);
    double predict(vector<double> &x);

private:
    Node *root;
    unsigned int max_depth;
    unsigned int lambda;
    double min_split_gain;
    double computeTerm(double g, double h, double lambda);
    double getSplitGain(double G, double H, double Gleft, double Hleft, double Gright, double Hright, double lambda);
    double getLeafWeight(double G, double H, double lambda);
    void fit(vector<vector<double>> &X, vector<double> &grad, vector<double> &hessian, Node *node, unsigned int current_depth);
    double predict(vector<double> &x, Node *node);
};

#endif
