#include <iostream>
#include <string>
#include <vector>

#include "lib.h"
#include "NeuNet.h"

using namespace std;

int main(int argc, char** argv) {

	srand(time(NULL));

	string file_name;
	char sep;
	bool skip;
	int n_col = 2, col_y = 1;

	file_name = argv[1];
	sep = argv[2][0];
	skip = stoi(argv[3]);
	if (argc >= 5) {
		n_col = stoi(argv[4]);
		col_y = stoi(argv[5]);
	}

	cout << "file: " << file_name << "\nsep: " << sep << "\nskip: " << skip << endl;
	vector<vector<double>> raw_data = data_extr(file_name, sep, skip);
	cout << "imported" << endl;
	print_vvd(raw_data);

	vector<vector<double>> norm_data = normalized(raw_data);
	cout << "normalized" << endl;
	print_vvd(norm_data);


	double lr, n, mgs, lim;
	cout << "learning rate (es.:0.01):	"; cin >> lr;
	cout << "number of iteration (es.:10e4):	"; cin >> n;
//	cout << "dd limit (es.:1e-10) "; cin >> lim;
	cout << "minigroup size (es.:0/15):	"; cin >> mgs;
	cout << "lr=" << lr << endl;
	cout << "n=" << n << endl;
	cout << "mgs=" << mgs << endl;
//	cout << "lim=" << lim << endl;
//
	vector<int> midlay;
	midlay.push_back(2);
	net A(n_col-1, 1, midlay, 0, 0, Sigmoid, dSigmoid, lr, true);	

	vector<vector<double>> data=raw_data;

	vector<vector<double>> in, out;
	for(size_t i=0; i<data.size(); i++){
		vector<double> t_i, t_o;
		for (size_t j = 0; j < n_col; j++) {
			
			if(j==col_y)
				t_i.push_back(data[i][j]);
			else
				t_o.push_back(data[i][j]);
		}
		in.push_back(t_i);
		out.push_back(t_o);
	}

	print_vvd(in);
	print_vvd(out);

	ofstream fout("cost.dat");
	for(int i=0; i<n; i++){
		vector<vector<double>> temp_in, temp_out;
		int t;
		for(int j=0; j<mgs; j++){
			t=double(rand())/RAND_MAX*in.size();
			temp_in.push_back(in[t]);
			temp_out.push_back(out[t]);
		}
		fout << i << ",	" << A.train(temp_in, temp_out) << endl;
		cout << i << ",	" << A.train(temp_in, temp_out) << endl;
	}

	fout.close();
	fout.open("plt");
	fout << "set terminal png medium size 640,480" << endl;
	fout << "set output 'loss.png'" << endl;
	fout<<"plot 'cost.dat' u 1:2 w l"<<endl;
	fout.close();

	system("gnuplot plt -p");

	cout<<A.test(in, out)<<endl;

	fout.open("pred.dat");
	for (size_t i = 0; i < data.size(); i++) {
		vector<double> net_o=A.calc(in[i]);
		fout << i << ",	" << in[i][0] << ",	" << in[i][1] << ",	" << out[i][0] << ",	" << net_o[0] << endl;
	}	
	fout.close();
	fout.open("plt2");
	fout << "set terminal png medium size 640,480" << endl;
	fout << "set output 'pred.png'" << endl;
	fout << "plot 'pred.dat' u 4:1 w p, 'pred.dat' u 5:1 w p" << endl;
	fout.close();
	system("gnuplot plt2 -p");

	
	return 0;
}