#include <iostream>
#include <fstream>
#include <stdlib.h>

using namespace std;

int main(int argc, char** argv) {

	if (argc != 4) {
		cerr << "incorrect # of arguments";
		exit(1);
	}

	ifstream mfile1(argv[1]);
	ifstream mfile2(argv[2]);
	ofstream mfile3(argv[3]);

	int Am;
	int An;

	mfile1 >> Am;
	mfile1 >> An;

	int Bm;
	int Bn;

	mfile2 >> Bm;
	mfile2 >> Bn;

	int A[Am][An];
	int B[Bm][Bn];
	int C[Am][Bn];

	for (int i = 0; i < Am; i++) {
		for (int j = 0; j < An; j++) {
			mfile1 >> A[i][j];
		}
	}

	for (int i = 0; i < Bm; i++) {
		for (int j = 0; j < Bn; j++) {
			mfile2 >> B[i][j];
		}
	}

	for (int i = 0; i < Am; i++) {
		for (int j = 0; j < Bn; j++) {
			C[i][j] = 0;
			for (int k = 0; k < An; k++) {
				C[i][j] = A[i][k]*B[k][j] + C[i][j];
			}
		}
	}

	mfile3 << Am << ' ' << Bn << endl;

	for (int i = 0; i < Am; i++) {
		for (int j = 0; j < Bn; j++) {
			mfile3 << C[i][j] << ' ';
		}
		mfile3 << endl;
	}
	return 0;
}
