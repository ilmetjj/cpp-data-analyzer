#include "NeuNet.h"

using namespace std;

double Relu(double x){
	if(x >= 0)
		return x;
	else
		return 0;
}
double dRelu(double x)
{
	if (x >= 0)
		return 1;
	else
		return 0;
}

double Softp(double x)
{
	if (x > -16 && x < 16)
		return log(exp(x) + 1);
	else if (x >= 16)
		return x;
	else
		return 0;
}
double dSoftp(double x)
{
	if (x > -16 && x < 16)
		return exp(x) / (exp(x) + 1);
	else if (x >= 16)
		return 1;
	else
		return 0;
}

double Sigmoid(double x){
	if (x > -16 && x < 16)
		return exp(x) / (exp(x) + 1);
	else if (x >= 16)
		return 1;
	else
		return 0;
}
double dSigmoid(double x)
{
	if (x > -16 && x < 16)
		return exp(x) / pow(exp(x) + 1, 2);
	else if (x >= 16)
		return 0;
	else
		return 0;
}

double Dir(double x){
	return x;
}
double dDir(double x){
	return 1;
}

/*NEURON*/

neuron::neuron(int _id, int n_prev, int n_post, double _w, double _b, double (*_act)(double), double (*_dact)(double), double _lr)
:id(_id), npr(n_prev), npst(n_post), b(_b), act(_act), dact(_dact), lr(_lr)
{
	w.resize(npr);
	prev.resize(npr);
	post.resize(npst);

	bool wi=(_w!=0);
	for(int i=0; i<npr; i++){
		if(wi){
			w[i] = wi * double(rand()) / RAND_MAX;
		}
		else{
			w[i]=double(rand())/RAND_MAX*2-1;
		}
	}

	cout<<"id: "<<id<<flush;
	cout<<" npr: "<<npr<<" npst: "<<npst<<" b: "<<b<<flush;
	cout<<" wsiz: "<<w.size()<<flush;
	for(int i=0; i<w.size(); i++){
		cout<<" w"<<i<<": "<<w[i]<<flush;
	}
//	cout << " act: " << act(-10) << " " <<act(-5) << " " << act(-1) << " " << act(0.5) << " " << act(0) << " " << act(0.5) << " " << act(1) << " " << act(5) << " " << act(10) <<flush;
//	cout << " dact: " << dact(-10) << " " << dact(-5) << " " << dact(-1) << " " << dact(0.5) << " " << dact(0) << " " << dact(0.5) << " " << dact(1) << " " << dact(5) << " " << dact(10) << flush;
	cout << " lr: " << lr << endl;
}

void neuron::calc(){
	z = b;
	for (int i = 0; i < npr; i++)
	{
		z+=(*prev[i]).a * w[i];
	}
	a = act(z);
}

void neuron::calc(vector<double> in){
	z = b;
	for (int i = 0; i < in.size(); i++)
	{
		z += in[i] * w[i];
	}
	a = act(z);
}

void neuron::back_pr(){
	dda=0;
	for (int i = 0; i < npst; i++)
	{
		dda += (*post[i]).ddp[id] * w[i];
	}

	vector<double> ddw;
	for (int i = 0; i < npr; i++)
	{
		ddw.push_back(dda*dact(z)*(*prev[i]).a);
	}
	grw.push_back(ddw);
	
	
	grb.push_back(dda*dact(z));

	ddp.clear();
	for (int i = 0; i < npr; i++)
	{
		ddp.push_back(dda * dact(z) * w[i]);
	}
}

void neuron::back_pr(double exp_y){
	dda = 2*(a-exp_y);

	vector<double> ddw;
	for (int i = 0; i < npr; i++)
	{
		ddw.push_back(dda*dact(z)*(*prev[i]).a);
	}
	grw.push_back(ddw);
	
	grb.push_back(dda*dact(z));
	for (int i = 0; i < npr; i++)
	{
		ddp.push_back(dda * dact(z) * w[i]);
	}
}

void neuron::back_pr(vector<double> in){
	dda = 0;
//	cout<<"cicl"<<endl;
	for (int i = 0; i < npst; i++)
	{
//		cout << post.size() << endl;
		dda += (*post[i]).ddp[id] * w[i];
	}
//	cout<<dda<<endl;

	vector<double> ddw;
	for (int i = 0; i < npr; i++)
	{
		ddw.push_back(dda * dact(z) * in[i]);
	}
	grw.push_back(ddw);
//	cout<<"ddw"<<endl;

	grb.push_back(dda * dact(z));
//	cout<<"b"<<endl;

	ddp.clear();
	for (int i = 0; i < npr; i++)
	{
		ddp.push_back(dda * dact(z) * w[i]);
	}
//	cout<<"ddp"<<endl;
}

void neuron::appl_gr(){
	vector<double> sgrw;
	sgrw.resize(npr, 0);

	for(int i=0; i<grw.size(); i++){
		for(int j=0; j<grw[i].size();j++){
			sgrw[j]+=grw[i][j];
		}
	}

	for (int i = 0; i < w.size(); i++)
	{
		sgrw[i] = sgrw[i] / grw.size() * lr;
		w[i]-=sgrw[i];
	}

	grw.clear();

	double sgrb;
	sgrb=0;

	for(int i=0; i<grb.size(); i++){
		sgrb+=grb[i];
	}

	sgrb = sgrb/ grb.size() *lr;
	b-=sgrb;

	grb.clear();
}

void neuron::set_post(vector<neuron> &_post) {
//	cout<<npst<<" "<<post.size()<<endl;
	for(int i=0; i<npst; i++)
	{
		post[i]=(&_post[i]);
	}
}
void neuron::set_prev(vector<neuron> &_prev){
	for (int i = 0; i < npr; i++)
	{
		prev[i]=(&_prev[i]);
	}
}
double neuron::read_val() { return a; }

/*NET*/

net::net(int n_input, int n_output, vector<int> n_hid, double w, double b, double (*_act)(double), double (*_dact)(double), double _lr, bool last_act)
:n_in(n_input), n_out(n_output)
{
	n_hid.push_back(n_out);
	
	vector<neuron> first;
	for (int i = 0; i < n_hid[0]; i++)
	{
		first.push_back(neuron(i, n_in, n_hid[1], w, b, _act, _dact, _lr));
	}
	lay.push_back(first);

	for (int i = 1; i < n_hid.size()-1; i++)
	{
		vector<neuron> temp;
		for (int j = 0; j < n_hid[i]; j++)
		{
			temp.push_back(neuron(j, n_hid[i-1], n_hid[i+1], w, b, _act, _dact, _lr));
		}
		lay.push_back(temp);
	}

	vector<neuron> last;
	for (int i = 0; i < n_out; i++)
	{
		if(last_act)
			last.push_back(neuron(i, n_hid[n_hid.size()-2], 1, w, b, _act, _dact, _lr));
		else
			last.push_back(neuron(i, n_hid[n_hid.size() - 2], 1, w, b, Dir, dDir, _lr));
	}
	lay.push_back(last);

	for (int i = 0; i < lay.size() - 1; i++)
	{
		for (int j = 0; j < lay[i].size(); j++)
		{
			lay[i][j].set_post(lay[i + 1]);
		}
	}

	for (int i = 1; i < lay.size(); i++)
	{
		for (int j = 0; j < lay[i].size(); j++)
		{
			lay[i][j].set_prev(lay[i - 1]);
		}
	}
}
vector<double> net::calc(vector<double> _in)
{
	in=_in;

//	cout<<"0"<<endl;
	for (int i = 0; i < lay[0].size(); i++)
	{
//		cout<<i<<": "<<flush;
		lay[0][i].calc(in);
	}
//	cout<<endl;

	for (int i = 1; i < lay.size(); i++)
	{
//		cout<<i<<endl;
		for (int j = 0; j < lay[i].size(); j++)
		{
//			cout<<j<<": "<<flush;
			lay[i][j].calc();
		}
//		cout<<endl;
	}

	vector<double> v;
	int l = lay.size() - 1;
	for (int i = 0; i < lay[l].size(); i++)
	{
		double a = lay[l][i].read_val();
		v.push_back(a);
//		cout<<i<<": "<<a<<", "<<flush;
	}
//	cout<<endl;
	return v;
}
void net::back_pr(vector<double> exp_y)
{
	int l = lay.size() - 1;
//	cout<<l<<endl;
	for (int i = 0; i < lay[l].size(); i++)
	{
//		cout<<i<<": "<<flush;
		lay[l][i].back_pr(exp_y[i]);
	}
//	cout<<endl;
	for (int i = l - 1; i > 0; i--)
	{
//		cout << i << endl;
		for (int j = 0; j < lay[i].size(); j++)
		{
//			cout << j << ": " << flush;
			lay[i][j].back_pr();
		}
//		cout<<endl;
	}
//	cout << "0" << endl;
	for (int i = 0; i < lay[l].size(); i++)
	{
//		cout << i << ": " << flush;
		lay[0][i].back_pr(in);
	}
//	cout<<endl;
}
void net::appl_gr()
{
	for (int i = 1; i < lay.size(); i++)
	{
		for (int j = 0; j < lay[i].size(); j++)
		{
			lay[i][j].appl_gr();
		}
	}
}
double net::train(vector<vector<double>> _in, vector<vector<double>> exp_y)
{
	double avg=0;
	for (int i = 0; i < _in.size(); i++)
	{
		vector<double> temp = calc(_in[i]);
		back_pr(exp_y[i]);
		double tav = 0;
		for (int j = 0; j < n_out; j++)
		{
			tav += pow(exp_y[i][j] - temp[j], 2);
		}
		avg += tav / n_in;
	}
	appl_gr();
	return avg / _in.size();
}
double net::test(vector<vector<double>> _in, vector<vector<double>> exp_y)
{
	double avg=0;
	for (int i = 0; i < _in.size(); i++)
	{
		vector<double> temp=calc(_in[i]);
		double tav=0;
		for (int j = 0; j < n_out; j++)
		{
			tav += pow(exp_y[i][j] - temp[j], 2);
		}
		avg+=tav/n_in;
	}
	return avg / _in.size();
}