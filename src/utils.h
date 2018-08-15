#ifndef UTILS_H
#define UTILS_H

#include <ctime>
#include <random>
#include <vector>

using std::vector;

class Utils
{
public:
    static bool same(double a, double b);
    static vector<double> subtract(const vector<double> &a, const vector<double> &b);
    static vector<double> square(const vector<double> &a);
    static vector<double> multiply(const vector<double> &a, int num);
    static double sum(const vector<double> &a);
    static double mean(const vector<double> &a);
    static vector<int> argsort(const vector<double> &a);
    static vector<double> getColumn(const vector<vector<double>> &a, int index);
    static vector<vector<double>> slice(const vector<vector<double>> &a, const vector<int> &idx);
    static vector<double> slice(const vector<double> &a, const vector<int> &idx);
    static vector<double> random(int size);
};

#endif
