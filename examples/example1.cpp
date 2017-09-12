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
#include <stdio.h>
#include <cstdlib>
#include <string.h>
#include <iostream>
#include <chrono>
#include <ctime>


#include <iostream>
using namespace std;

#include "RAJA/RAJA.hpp"
#include "RAJA/util/defines.hpp"

const int NN      = 20000000;
const int maxIter = 50;


RAJA_INLINE void simple_native(double *RAJA_RESTRICT I, double * A, double * B){
  
#pragma simd
  for(int id=0; id<NN; ++id){
    I[id] += 0.24*A[id] + 0.35*B[id];
  };
  
}


template<typename jacobiPolicy>
RAJA_INLINE void simple(double *RAJA_RESTRICT I, double * A, double * B){
  
  RAJA::forall<jacobiPolicy>
    (0,NN, [=] (int id) {      
      I[id] += 0.24*A[id] + 0.35*B[id];
    });    
}



int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{


  int iter=0;
  size_t dataSz = NN*sizeof(double);
  double *RAJA_RESTRICT I     = (double *) _mm_malloc(dataSz,64);
  double *RAJA_RESTRICT A     = (double *) _mm_malloc(dataSz,64);
  double *RAJA_RESTRICT B     = (double *) _mm_malloc(dataSz,64);

  //double *RAJA_RESTRICT I     = (double *) malloc(dataSz);
  //double *RAJA_RESTRICT Iold  = (double *) malloc(dataSz);

  memset(A, 0, NN * sizeof(double));
  memset(B, 0, NN * sizeof(double));
  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> elapsed_seconds;

  //-------[Sequential]--------
  start = std::chrono::system_clock::now();
  while(iter<maxIter){
    simple<RAJA::seq_exec>(I,A,B);
    iter++;
  }
  end = std::chrono::system_clock::now();

  elapsed_seconds = end-start;
  std::cout << "Seq elapsed time: " << elapsed_seconds.count() << "s\n";


  //Reset data
  iter = 0; 
  memset(A, 0, NN * sizeof(double));
  memset(B, 0, NN * sizeof(double));

  //-------[Native]--------
  start = std::chrono::system_clock::now();
  while(iter<maxIter){
    simple_native(I,A,B);
    iter++;
  }
  end = std::chrono::system_clock::now();

  elapsed_seconds = end-start;
  std::cout << "Native SIMD elapsed time: " << elapsed_seconds.count() << "s\n";


  //Reset data
  iter = 0; 
  memset(A, 0, NN * sizeof(double));
  memset(B, 0, NN * sizeof(double));


  //-------[SIMD]---------
  start = std::chrono::system_clock::now();
  while(iter<maxIter){
    simple<RAJA::simd_exec>(I,A,B);
    iter++;
  }
  end = std::chrono::system_clock::now();


  elapsed_seconds = end-start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "RAJA SIMD elapsed time: " << elapsed_seconds.count() << "s\n";


  _mm_free(I);
  _mm_free(A);
  _mm_free(B);
  return 0;
}
