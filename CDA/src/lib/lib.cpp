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

double Loss(vector<double> p, vector<vector<double>> v, int n_col, int col_y) {
	double s = 0;
	for (int i = 0; i < v.size(); i++)
	{
		double temp = v[i][col_y];
		for (int j = 0; j < n_col; j++) {
			if (j != col_y) {
				temp -= p[j] * v[i][j];
			}
			temp -= p[col_y];
			s+=pow(temp,2);
		}
	}
	return s / v.size();
}
double ddmLoss(vector<double> p, double col_x, vector<vector<double>> v, int mgs, int n_col, int col_y) {
	double s = 0;
	if (mgs == 0) {
		for (int i = 0; i < v.size(); i++)
		{
			double temp=v[i][col_y];
			for (int j = 0; j < n_col ; j++)
			{
				if(j!=col_y){
					temp-=p[j]*v[i][j];
				}
			}
			temp-=p[col_y];
			s -= 2 * v[i][col_x] * temp;
		}
		return s / v.size();
	}
	else {
		for (int i = 0; i < mgs; i++)
		{
			int h = int(double(double(rand()) / double(RAND_MAX)) * double(v.size()));
			double temp = v[h][col_y];
			for (int j = 0; j < n_col; j++)
			{
				if (j != col_y) {
					temp -= p[j] * v[h][j];
				}
			}
			temp -= p[col_y];
			s -= 2 * v[h][col_x] * temp;
		}
		return s / mgs;
	}
	return s;
}
double ddqLoss(vector<double> p, vector<vector<double>> v, int mgs, int n_col, int col_y){
	double s = 0;
	if (mgs == 0) {
		for (int i = 0; i < v.size(); i++)
		{
			double temp = v[i][col_y];
			for (int j = 0; j < n_col; j++)
			{
				if (j != col_y) {
					temp -= p[j] * v[i][j];
				}
			}
			temp -= p[col_y];
			s -= 2 * temp;
		}
		return s / v.size();
	}
	else {
		for (int i = 0; i < mgs; i++)
		{
			int h = int(double(double(rand()) / double(RAND_MAX)) * double(v.size()));
			double temp = v[h][col_y];
			for (int j = 0; j < n_col; j++)
			{
				if (j != col_y) {
					temp -= p[j] * v[h][j];
				}
			}
			temp -= p[col_y];
			s -= 2 * temp;
		}
		return s / mgs;
	}
	return s;
}

void linreg(vector<double>& p, vector<vector<double>> &data, double lr, int n, double lim, int mgs, int n_col, int col_y){
	double l = Loss(p, data, n_col, col_y);
	cout.precision(10);
	for (int i = 0; i < n; i++) {
		cout<<i<<endl;
		double l0 = l;
		vector<double> p0=p;
		vector<double> ddp;
		ddp.resize(n_col);
		for(int j=0; j<n_col; j++){
			if(j!=col_y){
				ddp[j]=ddmLoss(p,j,data,mgs,n_col,col_y);
			}
		}
		ddp[col_y]=ddqLoss(p,data,mgs,n_col,col_y);
		double lrn = lr; // (log10(double(i+1))+1);
		vector<double> stepp;
		stepp.resize(n_col);
		for (int j = 0; j < n_col; j++) {
			stepp[j]=lrn*ddp[j];
			p[j] = p0[j] - stepp[j];
		}
		l = Loss(p, data, n_col, col_y);
		double dl = l - l0;
		for(int j=0; j<n_col; j++){
			cout<<"dd"<<j<<"= "<<ddp[j]<<" "<<flush;
		}
		cout<<endl;
		for (int j = 0; j < n_col; j++) {
			cout << "step" << j << "= " << stepp[j] << " " << flush;
		}
		cout << endl;
		for (int j = 0; j < n_col; j++) {
			cout << "p" << j << "= " << p[j] << " " << flush;
		}
		cout << endl;
		cout << "loss= " << l << "		Dloss=" << dl << endl;
		bool min=true;
		for (int j = 0; j < n_col; j++) {
			if(abs(ddp[j])>lim)
				min=false;
		}
		if(min)
			break;
	}
}

void plot(string file_name, vector<double>& p, int n_col, int col_y){
	ofstream fout("plt");

	fout << "set terminal png medium size 640,480" << endl;

	for (size_t i = 0; i < n_col; i++)
	{
		if(i!=col_y)
			fout << "r"<<i<<"p(x)=x*" << p[i] << "+" << p[col_y] << endl;
	}
	
	for (size_t i = 0; i < n_col; i++)
	{
		
		if (i != col_y){
			fout << "set output 'graph" << i << ".png'" << endl;
			fout << "plot '" << file_name << "' u "<<i+1<<":"<<col_y+1<<" w p, r"<<i<<"p(x)" << endl;
		}
	}
	fout.close();
	system("gnuplot plt -p");
}

void plot(vector<vector<double>>data, vector<double>p, int n_col, int col_y, string file_name) {
	ofstream fout(file_name);

	for(size_t i=0; i<data.size(); i++){
		for (size_t j = 0; j < data[i].size(); j++)
		{
			fout<<data[i][j]<<", "<<flush;
		}
		fout<<endl;
	}

	fout.close();
	fout.open("plt");
	fout << "set terminal png medium size 640,480" << endl;

	for (size_t i = 0; i < n_col; i++)
	{
		if (i != col_y)
			fout << "r" << i << "p(x)=x*" << p[i] << "+" << p[col_y] << endl;
	}

	for (size_t i = 0; i < n_col; i++)
	{

		if (i != col_y) {
			fout << "set output 'graph" << i << ".png'" << endl;
			fout << "plot '" << file_name << "' u " << i + 1 << ":" << col_y + 1 << " w p, r" << i << "p(x)" << endl;
		}
	}
	fout.close();
	system("gnuplot plt -p");
}