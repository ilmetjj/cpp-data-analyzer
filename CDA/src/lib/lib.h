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

#endif
