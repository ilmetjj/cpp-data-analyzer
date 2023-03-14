#ifndef LIB_H
#define LIB_H

#include <sstream>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>

using namespace std;

vector<vector<double>> data_extr(string file_name, char sep=',', bool skip=false);
void print_vvd(vector<vector<double>> v);
//vector<double[2]> convert(vector<vector<double>> v);
vector<vector<double>> normalized(vector<vector<double>> raw);
double Loss(double m, double q, vector<vector<double>> data, int cx=0, int cy=1);
double ddmLoss(double m, double q, vector<vector<double>> data, int mgs=0, int cx=0, int cy=1);
double ddqLoss(double m, double q, vector<vector<double>> data, int mgs=0, int cx=0, int cy=1);
void plot(string file_name, double m, double q);

double Loss(vector<double> p, vector<vector<double>> data, int n_col = 2, int col_y = 1);
double ddmLoss(vector<double> p, double col_x, vector<vector<double>> data, int mgs = 0, int n_col =2, int col_y = 1);
double ddqLoss(vector<double> p, vector<vector<double>> data, int mgs = 0, int n_col = 2, int col_y = 1);

void linreg(vector<double>& p, vector<vector<double>> &data, double lr, int n, double lim, int mgs=0, int n_col=2, int col_y=1);
void plot(string file_name, vector<double>& p, int n_col, int col_y);
void plot(vector<vector<double>> data, vector<double> p, int n_col, int col_y, string file="dat");

#endif
