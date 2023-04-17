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
net::net(){}
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
			last.push_back(neuron(i, n_hid[n_hid.size() - 2], 1, w, b, _act, _dact, _lr));
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

//**********************************************************************************************************//

bool neuron::save(fstream& fout){
	fout.write(reinterpret_cast<char*>(&id), sizeof(id));
	fout.write(reinterpret_cast<char*>(&npr), sizeof(npr));
	fout.write(reinterpret_cast<char*>(&npst), sizeof(npst));
	fout.write(reinterpret_cast<char*>(&b), sizeof(b));
	int s = w.size();
	fout.write(reinterpret_cast<char*>(&s), sizeof(s));
	for (size_t i = 0; i < w.size(); i++){
		fout.write(reinterpret_cast<char*>(&w[i]), sizeof(w[i]));
	}
	
	return true;
}
bool neuron::load(fstream& fin){
	fin.read(reinterpret_cast<char*>(&id), sizeof(id));
	fin.read(reinterpret_cast<char*>(&npr), sizeof(npr));
	fin.read(reinterpret_cast<char*>(&npst), sizeof(npst));
	fin.read(reinterpret_cast<char*>(&b), sizeof(b));
	int s;
	fin.read(reinterpret_cast<char*>(&s), sizeof(s));
	w.resize(s);
	for (size_t i = 0; i < w.size(); i++) {
		fin.read(reinterpret_cast<char*>(&w[i]), sizeof(w[i]));
	}
	return true;
}

bool net::save(string file) {
	fstream fout;
	fout.open(file, ios::binary | ios::out);
//	fout.seekp(0, ios::beg);

	fout.write(reinterpret_cast<char*>(&n_in), sizeof(n_in));
	fout.write(reinterpret_cast<char*>(&n_out), sizeof(n_out));
	int s=lay.size();
	fout.write(reinterpret_cast<char*>(&s), sizeof(s));
	for(int i=0; i<s; i++){
		int t=lay[i].size();
		fout.write(reinterpret_cast<char*>(&t), sizeof(t));
		for (int j = 0; j < t; j++) {
			lay[i][j].save(fout);
		}
	}

	return true;
}
bool net::load(string file, double (*_act)(double), double (*_dact)(double), double _lr, bool last_act) {
	fstream fin;
	fin.open(file, ios::binary | ios::in);
	
	fin.read(reinterpret_cast<char*>(&n_in), sizeof(n_in));
	fin.read(reinterpret_cast<char*>(&n_out), sizeof(n_out));
	int s;
	fin.read(reinterpret_cast<char*>(&s), sizeof(s));
	lay.resize(s);
	for (int i = 0; i < s; i++) {
		int t;
		fin.read(reinterpret_cast<char*>(&t), sizeof(t));
		lay[i].resize(t);
		for (int j = 0; j < t; j++) {
			lay[i][j].load(fin);
			if(last_act or i<s-1)
			{
				lay[i][j].act = _act;
				lay[i][j].dact = _dact;
			}
			else
			{
				lay[i][j].act = Dir;
				lay[i][j].dact = dDir;
			}
			lay[i][j].lr = _lr;
		}
	}

	cout << "dafeaf" << endl;

	for (int i = 0; i < lay.size() - 1; i++)
	{
		for (int j = 0; j < lay[i].size(); j++)
		{
			lay[i][j].post.resize(lay[i + 1].size());
			lay[i][j].set_post(lay[i + 1]);
		}
	}

	for (int i = 1; i < lay.size(); i++)
	{
		for (int j = 0; j < lay[i].size(); j++)
		{
			lay[i][j].prev.resize(lay[i - 1].size());
			lay[i][j].set_prev(lay[i - 1]);
		}
	}

	cout<<"dafeaf"<<endl;

	return true;
}
//**********************************************************************************************************//
/*
	ifstream fin(file, ios::binary);

	uint8_t header[80];
	fin.seekg(0, ios::beg);
	fin.read((char*)&header, sizeof(header));
	cout << header << endl;
	uint32_t N_tr;
	fin.read((char*)&N_tr, sizeof(N_tr));
	cout << N_tr << endl;
	n_tr=N_tr;

	double X=0,x=0,Y=0,y=0,Z=0,z=0;
	for (size_t i = 0; i < N_tr; i++)
	{
		triangle t;
		_Float32 N[3], A[3], B[3], C[3];
		uint16_t end;
		fin.read((char*)&N, sizeof(N));
		fin.read((char*)&A, sizeof(A));
		fin.read((char*)&B, sizeof(B));
		fin.read((char*)&C, sizeof(C));
		fin.read((char*)&end, sizeof(end));
		t.n = vettore(N[0], N[1], N[2]);
		t.a = vettore(A[0], A[1], A[2])*scale;
		t.b = vettore(B[0], B[1], B[2])*scale;
		t.c = vettore(C[0], C[1], C[2])*scale;

*/