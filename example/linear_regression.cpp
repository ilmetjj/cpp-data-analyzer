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
	cout<<"lr="<<lr<<endl;
	cout<<"n="<<n<<endl;
	cout<<"mgs="<<mgs<<endl;
	cout<<"lim="<<lim<<endl;

	vector<double> p;
	p.resize(n_col,0);
	linreg(p, norm_data, lr, n, lim, mgs, n_col, col_y);
	plot(norm_data,p,n_col,col_y);

	return 0;
}
