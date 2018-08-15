#include <algorithm>
#include "utils.h"

using namespace std;

const double EPSINON = 0.000001;

bool Utils::same(double a, double b)
{
    double x = a - b;
    if ( (x >= -EPSINON) && (x <= EPSINON) )
        return true;
    return false;
}

vector<double> Utils::subtract(const vector<double> &a, const vector<double> &b)
{
    int len = a.size();
    vector<double> result(len);
    for (int i = 0; i < len; i++)
        result[i] = a[i] - b[i];
    return result;
}

vector<double> Utils::square(const vector<double> &a)
{
    int len = a.size();
    vector<double> result(len);
    for (int i = 0; i < len; i++)
        result[i] = a[i] * a[i];
    return result;
}

vector<double> Utils::multiply(const vector<double> &a, int num)
{
    int len = a.size();
    vector<double> result(a);
    for (int i = 0; i < len; i++)
        result[i] *= num;

    return result;
}

double Utils::sum(const vector<double> &a)
{
    int len = a.size();
    double sum = 0.;
    for (int i = 0; i < len; i++)
        sum += a[i];
    return sum;
}

double Utils::mean(const vector<double> &a)
{
    int len = a.size();
    double sum = Utils::sum(a);
    return sum / len;
}

vector<int> Utils::argsort(const vector<double> &a)
{
    int len = a.size();
    vector<int> idx(len);
    for (int i = 0; i < len; i++)
        idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int x, int y) { return a[x] < a[y];});

    return idx;
} 

vector<double> Utils::getColumn(const vector<vector<double>> &a, int index)
{
    int rows = a.size();
   
    vector<double> result;
    for (int i = 0; i < rows; i++)
        result.push_back(a[i][index]);

    return result;
}

vector<vector<double>> Utils::slice(const vector<vector<double>> &a, const vector<int> &idx)
{
    vector<vector<double>> result;
    for (auto i : idx)
    {
        result.push_back(a[i]);
    }

    return result;
}

vector<double> Utils::slice(const vector<double> &a, const vector<int> &idx)
{
    vector<double> result;
    for (auto i : idx)
    {
        result.push_back(a[i]);
    }

    return result;
}

vector<double> Utils::random(int size)
{
    unsigned int seed = time(NULL);
    default_random_engine generator(seed);
    uniform_real_distribution<double> distribution(0., 1.);
    vector<double> result;
    for (int i = 0; i < size; i++)
        result.push_back(distribution(generator));

    return result;
}
