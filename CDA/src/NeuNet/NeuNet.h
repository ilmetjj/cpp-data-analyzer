#ifndef __HEADER_H__
#define __HEADER_H__

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

using namespace std;

#define LR 1e-8

class neuron;
class net;

double Relu(double z);
double dRelu(double z);

double Softp(double z);
double dSoftp(double z);

double Sigmoid(double z);
double dSigmoid(double z);

double Dir(double z);
double dDir(double z);

class neuron{
private:
	int id, npr, npst;
	double a, z, b, lr, dda;
	vector<double> w, grb, ddp;
	vector<vector<double>> grw;
	vector<neuron*> prev, post;
	double (*act)(double);
	double (*dact)(double);

public:
	neuron(int _id=0, int n_prev=0, int n_post=0, double w=0, double b=0, double (*_act)(double)=Softp, double (*_dact)(double)=dSoftp, double _lr=LR);

	void calc();
	void calc(vector<double> in);
	void back_pr();
	void back_pr(double exp_y);
	void back_pr(vector<double> in);
	void appl_gr();

	void set_post(vector<neuron> &_post);
	void set_prev(vector<neuron> &_prev);
	inline double read_val();
};

class net{
private:
	vector<vector<neuron>> lay;
	int n_in, n_out;
	vector<double> in;

	void back_pr(vector<double> exp_y);
	void appl_gr();
public:
	net(int n_input, int n_output, vector<int> n_hid, double w=0, double b=0, double (*_act)(double)=Softp, double (*_dact)(double)=dSoftp, double _lr=LR, bool last_act=true);
	
	vector<double> calc(vector<double> in);

	double train(vector<vector<double>> in, vector<vector<double>> exp_y);
	double test(vector<vector<double>> in, vector<vector<double>> exp_y);
};


#endif

