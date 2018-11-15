/*
   Copyright (c) 2010-2015, Ardavan Pedram and The University of Texas at Austin
   All rights reserved.

   This file is part of LAP_SIM and is under the BSD 3-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-3-Clause
*/

#include "Parameters.h"
#include <iostream>
#include <fstream>
#include "stdlib.h"
using namespace std;

void dump_matrix_file(int rows, int columns, int** matrix_A, ofstream& A_file);
void make_test_matrix(int rows, int columns, int** test_matrix, int preset, Test_Matrix type);

int number_of_rows,number_of_columns;
int matrix_type;
string test_file;
int ** test_matrix;
int preset;

ofstream Out_file;

int main(int argc, char* argv[] ) {
	cout << "!!!Linear Algebra Processor Matrix Maker !!!" << endl; // prints !!!Hello World!!!

	cout<<	"0: Diagonal with fixed preset"<<endl<<
			"1: Increasing diagonal with Preset"<<endl<<
			"2: Increasing diagonal with multiple of preset"<<endl<<
			"3: All 0 matrix"<<endl<<
			"4: All 1 matrix"<<endl<<
			"5: Constant Matrix with preset value"<<endl<<
			"6: Top Right aligned triangular with preset"<<endl<<
			"7: Top Left aligned triangular with Preset	"<<endl<<
			"8:Lower triangular with preset"<<endl<<
		    "9: Banded Matrix"<<endl <<
      "10: Row Major Ascending"<<endl;


	cout<<"Matrix Type?"<<endl;
	cin>>matrix_type;

	cout<<"Matrix preset constant?"<<endl;
	cin>>preset;


	cout<< "Enter Matrix Dimensions:"<<endl;
	cout<< "Matrix rows?"<<endl;
	cin>> number_of_rows;

	cout<< "Matrix columns?"<<endl;
	cin>> number_of_columns;

	cout<<"Ouput file?"<<endl;
	cin>>test_file;

	Out_file.open(test_file.c_str());
	if (! (Out_file.is_open()))
  {
    cout<<"Error, failed to open file "<< test_file<<endl;
    return 0;
  }

	test_matrix=(int **)malloc(sizeof(int *) * (number_of_rows));

	make_test_matrix(number_of_rows, number_of_columns, test_matrix, preset, (Test_Matrix) matrix_type);
	dump_matrix_file(number_of_rows,  number_of_columns, test_matrix, Out_file);
	cout<<"Matrix is dumped into file "<<test_file<<endl;
}

void make_test_matrix(int rows, int columns, int ** test_matrix, int preset, Test_Matrix type){

/*
	types:

	DIAG                  : Diagonal with fixed preset
	DIAG_INCR             : Increasing diagonal with Preset
	DIAG_MULT             : Increasing diagonal with multiple of preset
	ALL0                  : All 0 matrix
	ALL1                  : All 1 matrix
    PRESET                : Constant Matrix with preset value
	UP_RIGHT_TRIU_PRESET  : Top Right aligned triangular with preset
	UP_LEFT_TRIU_PRESET   : Top Left aligned triangular with Preset
    LO_TRIU_PRESET        : Lower triangular with preset
    BANDED                : Banded Matrix

*/
	int i,j;

  if (type==DIAG) {//  diagonal with fixed preset
		for ( i=0; i<rows; i++){
			test_matrix[i]=(int *)  malloc ( sizeof (int ) * (columns));
			for (j=0; j<columns; j++)
        if (i==j) test_matrix[i][j]=preset;
				else test_matrix[i][j]=0;
		}
	}

  else if (type==DIAG_INCR){  // increasing diagonal
		for ( i=0; i<rows; i++){
			test_matrix[i]=(int*)  malloc ( sizeof (int) * (columns));
			for (j=0; j<columns; j++)
        if (i==j) test_matrix[i][j]=preset+i;
				else test_matrix[i][j]=0;
		}
  }

  else if (type==DIAG_MULT){ // increasing diagonal with multiple of preset
		for ( i=0; i<rows; i++){
			test_matrix[i]=(int*)  malloc ( sizeof (int) * (columns));
			for (j=0; j<columns; j++)
        if (i==j) test_matrix[i][j]=preset* (i+1);
				else test_matrix[i][j]=0;
		}
  }


  else if (type==ALL1) {  //ALL 1
		for ( i=0; i<rows; i++){
			test_matrix[i]=(int*)  malloc ( sizeof (int) * (columns));
			for (j=0; j<columns; j++)
			  test_matrix[i][j]=1;
		}
	}

	else if (type==ALL0) {  // ALL 0
		for ( i=0; i<rows; i++){
			test_matrix[i]=(int*)  malloc ( sizeof (int) * (columns));
      //if (rows%NumofKernel==0 && i!=0) preset++;
			for (j=0; j<columns; j++)
				test_matrix[i][j]= 0;
		}

	}

	else if (type==PRESET) {  // ALL 0
		for ( i=0; i<rows; i++){
			test_matrix[i]=(int*)  malloc ( sizeof (int) * (columns));
      //if (rows%NumofKernel==0 && i!=0) preset++;
			for (j=0; j<columns; j++)
				test_matrix[i][j]= preset;
		}

	}

  else if (type==UP_LEFT_TRIU_PRESET) { //Upper Triangular left aligned
		for ( i=0; i<rows; i++){
			test_matrix[i]=(int*)  malloc ( sizeof (int) * (columns));
      //if (rows%NumofKernel==0 && i!=0) preset++;
			for (j=0; j<columns; j++){
          if (rows-i>j ) test_matrix[i][j] = preset;
					else test_matrix[i][j]=0;
      }
		}
  }

  else if (type==UP_RIGHT_TRIU_PRESET) { //Upper Triangular Right Alighned with preset
			for ( i=0; i<rows; i++){
				test_matrix[i]=(int*)  malloc ( sizeof (int) * (columns));
				for (j=0; j<columns; j++){
          if (j-rows>i ) test_matrix[i][j] = preset;
					else test_matrix[i][j]=0;
				}
			}
		}

  else if (type==LO_TRIU_PRESET) { //lower triangular with preset
			for ( i=0; i<rows; i++){
				test_matrix[i]=(int*)  malloc ( sizeof (int) * (columns));
				for (j=0; j<columns; j++){
          if (j-i<=0 ) test_matrix[i][j] = preset;
					else test_matrix[i][j]=0;
				}
			}
		}

	else if (type==BANDED) {// banded matrix
		for ( i=0; i<rows; i++){
			test_matrix[i]=(int*)  malloc ( sizeof (int) * (columns));
			for (j=0; j<columns; j++){
				//test_matrix[i][j]=(preset*100)+i+ .01*j;
				if (i==j) test_matrix[i][j]=(int)(preset+0.01*i+.0001*j);
				else if (i==(j-1)) test_matrix[i][j]=(int)(preset+0.01*i+.0001*j);
				else if (i==(j+1)) test_matrix[i][j]=(int)(preset+0.01*i+.0001*j);
				else test_matrix[i][j]=0;
			}
		}
	}

  else if (type == RM_ASCENDING) {
		for ( i=0; i<rows; i++){
			test_matrix[i]=(int*)  malloc ( sizeof (int) * (columns));
			for (j=0; j<columns; j++){
				test_matrix[i][j]= j + (i * columns);
			}
		}
    
  }

}

void dump_matrix_file(int rows, int columns, int** matrix_A, ofstream& A_file){

	int i,j;

	A_file<<rows<<" "<<columns<<endl;

	for (i=0;i<rows;i++)
	{
		for (j=0;j<columns;j++)
		{
			A_file << fixed << left << matrix_A[i][j] << " ";
		}
    A_file<<endl;
	}

	A_file<<endl<<"****************the end****************************"<<endl;
	A_file.close();

}

