#include <iostream>
#include <string>
#include <vector>

#include "lib/lib.h"

using namespace std;

int main(int argc, char** argv){

	srand(time(NULL));

	string file_name;
	char sep;
	bool skip;
	int n_col=2, col_y=1;

	file_name=argv[1];
	sep=argv[2][0];
	skip=stoi(argv[3]);
	if(argc>=5){
	n_col=stoi(argv[4]);
	col_y=stoi(argv[5]);
	}

/*	
	cout<<"data file (es.: file.csv):	"; cin>>file_name;
	cout<<"separator (es.: ,):	"; cin>>sep;
	cout<<"skip first line (0/1):	"; cin>>skip;
*/	
	cout<<"file: "<<file_name<<"\nsep: "<<sep<<"\nskip: "<<skip<<endl;
	vector<vector<double>> raw_data=data_extr(file_name, sep, skip);
	cout<<"imported"<<endl;
	print_vvd(raw_data);

	vector<vector<double>> norm_data=normalized(raw_data);
	cout<<"normalized"<<endl;
	print_vvd(norm_data);

	double lr, n, mgs, lim;
	cout<<"learning rate (es.:0.01):	"; cin>>lr;
	cout<<"number of iteration (es.:10e5):	"; cin>>n;
	cout<<"dd limit (es.:1e-10) "; cin>>lim;
	cout << "minigroup size (es.:0/15):	"; cin >> mgs;
	//	vector<double[2]> d=convert(data);
	cout<<"lr="<<lr<<endl;
	cout<<"n="<<n<<endl;
	cout<<"mgs="<<mgs<<endl;
	cout<<"lim="<<lim<<endl;

	vector<double> p;
	p.resize(n_col,0);
	linreg(p, norm_data, lr, n, lim, mgs, n_col, col_y);
	plot(norm_data,p,n_col,col_y);

/*
	double m=0;
	double q=0;
	double l=Loss(q,m,data,0,1);
	cout.precision(10);
	for(int i=0; i<n; i++){
		double m0=m, q0=q, l0=l;
		double ddm=ddmLoss(m0,q0,data,mgs,0,1), ddq=ddqLoss(m0,q0,data,mgs,0,1);
		double lrn = lr; // (log10(double(i+1))+1);
		double stepm = ddm * lrn, stepq = ddq * lrn;
		m=m0-stepm;
		q=q0-stepq;
		l=Loss(m,q,data,0,1);
		double dl=l-l0;
		cout << "ddm= " << ddm << "	ddq= " << ddq << "	lr= " << lrn << endl;
		cout << i << ":" << endl;
		cout << "stepm= " << stepm << "		stepq= " << stepq << endl;
		cout << "m= " << m << "		q= " << q << endl;
		cout << "r: y=" << m << "*x+" << q << endl;
		cout << "loss= " << l << "		Dloss=" << dl << endl;
		if(abs(ddm)<lim && abs(ddq)<lim)
			break;
	}

	plot(file_name,m,q);
*/	
	return 0;
}
