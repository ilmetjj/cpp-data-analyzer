#include "lib.h"

using namespace std;

vector<vector<double>> data_extr(string file_name, char sep, bool skip)
{
	vector<vector<double>> vv;
	ifstream fin(file_name);
	string line;
	if (skip)
		getline(fin, line);

	while (getline(fin, line))
	{
		vector<double> v;
		stringstream s(line);
		string val;
		while (getline(s, val, sep))
		{
			v.push_back(stod(val));
		}
		vv.push_back(v);
	}
	return vv;
}

void print_vvd(vector<vector<double>> v)
{
	for (int i = 0; i < v.size(); i++)
	{
		cout<<i<<":	"<<flush;
		for (int j = 0; j < v[i].size(); j++)
		{
			cout<< v[i][j] << "	"<<flush;
		}
		cout << endl;
	}
}
/*
vector<double[2]> convert(vector<vector<double>> v){
	vector<double[2]> w;
	for (int i = 0; i < v.size(); i++)
	{
		double a[2];
		v[i][0]=a[0];
		v[i][1]=a[1];
		w.push_back(a);
	}
	return w;
}
*/

vector<vector<double>> normalized(vector<vector<double>> raw){
	vector<double> max, min;
	max.resize(raw[0].size(), 0);
	min.resize(raw[0].size(), 0);
	for(int i=0; i<raw[0].size(); i++){
		for(int j=0; j<raw.size(); j++){
			if(raw[j][i]<min[i]){
				min[i]=raw[j][i];
			}
			if(raw[j][i]>max[i]){
				max[i]=raw[j][i];
			}
		}
	}
	for(int i=0; i<raw[0].size(); i++){
		for(int j=0; j<raw.size(); j++){
			raw[j][i]=(raw[j][i]-min[i])/(max[i]-min[i]);
		}
	}
	return raw;
}

double Loss(double m, double q, vector<vector<double>> v, int cx, int cy)
{
	double s=0;
	for (int i = 0; i < v.size(); i++)
	{
		s += pow((v[i][cy] - (m * v[i][cx] + q)), 2);
	}
	return s/v.size();
}
double ddmLoss(double m, double q, vector<vector<double>> v, int mgs, int cx, int cy)
{

	double s = 0;
	if(mgs==0){
		for (int i = 0; i < v.size(); i++)
		{
			s += -2 * v[i][cx] * (v[i][cy] - (m * v[i][cx] + q));
		}
		return s/v.size();
	}
	else{
		for (int i = 0; i < mgs; i++)
		{
			int j = int(double(double(rand()) / double(RAND_MAX)) * double(v.size()));
			s += -2 * v[j][cx] * (v[j][cy] - (m * v[j][cx] + q));
		}
		return s/mgs;
	}
	return s;
}
double ddqLoss(double m, double q, vector<vector<double>> v, int mgs, int cx, int cy)
{
	double s = 0;
	if(mgs==0){
		for (int i = 0; i < v.size(); i++)
		{
			s += -2 * (v[i][cy] - (m * v[i][cx] + q));
		}
		return s/v.size();
	}
	else{
		for (int i = 0; i < mgs; i++)
		{
			int j = int(double(double(rand()) / double(RAND_MAX)) * double(v.size()));
			s += -2 * (v[j][cy] - (m * v[j][cx] + q));
		}
		return s/mgs;
	}
	return s;
}

void plot(string file_name, double m, double q){
	ofstream fout("plt");
	fout << "r(x)=x*" << m << "+" << q << endl;
	fout << "plot '" << file_name << "' u 1:2 w p, r(x)" << endl;
	fout.close();
	system("gnuplot plt -p");
}