//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <iostream>

#include "RAJA/RAJA.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/index/IndexSetBuilders.hpp"

//Where is this?

/*Sparse Matrix
  A = [3 0 1 0]
      [0 0 0 0]
      [0 2 4 1]
      [1 0 0 1]
 */

using namespace RAJA;
void setup(double *A,double *x, double *col_index, double *row_ptr){

  //Populate Matrix
  A[0] = 3; A[1] = 1; A[2] = 2; A[3] = 4;  
  A[4] = 1; A[5] = 1; A[6] = 1;
  
  //Vector to multiply with
  x[0] = 1; x[1] = 1;  
  x[2] = 1; x[3] = 1;
  
  //populate column index
  col_index[0] = 0;  col_index[1] = 2; col_index[2] = 1;  col_index[3] = 2;
  col_index[4] = 3;  col_index[5] = 0; col_index[6] = 3;

  //Row pointers
  row_ptr[0] = 0; row_ptr[1] = 2; row_ptr[2] = 2;  
  row_ptr[3] = 5; row_ptr[4] = 7;  
}



//Sparse Matrix-Vector Multiply using CSR format
int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  std::cout<<"Example 5: Sparse Matrix-Vector Multipy"<<std::endl; 

  int n       = 4; //A is an nxn matrix
  int nzr_ct  = 7; //number of non-zeros in A
  int cInd_ct = 7; //column index
  int rPtr_ct = 5; //Row pointers

  double *A      = new double[nzr_ct];
  int *col_index = new int[cInd_ct];
  int *row_ptr   = new int[rPtr_ct];

  double *x         = new double[n];
  double *y         = new double[n];

 
  //----[C-Style Loop]-----------
  for(int row=0; row<n; ++row){    
    double dot = 0.0;
    int row_start = row_ptr[row];
    int row_end   = row_ptr[row+1];    
    for(int elem = row_start; elem < row_end; ++elem){
      dot += A[elem]*x[col_index[elem]];
    }    
    y[row] = dot;
  }
  //==============================

#if 0
  RAJA::IndexSet myIndexSet;
  int elems [] = {0,1,2,3,4,5,6,7};
  RAJA::buildIndexSetAligned(myIndexSet,col_index,cInd_ct); 

  RAJA::forall<RAJA::seq_exec>(myIndexSet,[=](int i){
      std::cout<<i<<std::endl;
    });
#endif
  



  

#if 0
  //Introduce IndexSet    
  //----[RAJA-Style Execution]----
  std::cout<<"RAJA: Squential Policy"<<std::endl;
  RAJA::forall<RAJA::seq_exec>(0,n,[=](int row){

      double dot    = 0.0;
      int row_start = row_ptr[row];
      int row_end   = row_ptr[row+1];
      for(int elem = row_start; elem < row_end; ++elem){
        dot += A[elem]*x[col_index[elem]];
      }
      y[row] = dot; 
    }
  //==============================
#endif

  for(int i=0; i<n; ++i){
    std::cout<<y[i]<<std::endl;
  }

  



  delete[] A, cInd_ct, rPtr_ct;


  return 0;
}
