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

const int N       = 1024;
const int NN      = (N + 2) * (N + 2);
const int maxIter = 1000;


template<typename jacobiPolicy>
void Jacobi(double *RAJA_RESTRICT I, double *RAJA_RESTRICT Iold){

  RAJA::forall<RAJA::seq_exec>
    (1,N+1, [=] (int n) {
      
      RAJA::forall<jacobiPolicy>
        (1,N+1, [=] (int m) {
          
          int id = n * (N + 2) + m;         
          I[id] = - 0.25 * (Iold[id - N - 2] + Iold[id + N + 2] + Iold[id - 1] + Iold[id + 1]);          
        });
    });

  RAJA::forall<jacobiPolicy>(
    0,NN, [=] (int k) {      
        Iold[k] = I[k];
  });
    
}

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{


  int iter=0;
  size_t dataSz = NN*sizeof(double);
  double *RAJA_RESTRICT I     = (double *) _mm_malloc(dataSz,64);
  double *RAJA_RESTRICT Iold  = (double *) _mm_malloc(dataSz,64);

  //double *RAJA_RESTRICT I     = (double *) malloc(dataSz);
  //double *RAJA_RESTRICT Iold  = (double *) malloc(dataSz);

  memset(I, 0, NN * sizeof(double));
  memset(Iold, 0, NN * sizeof(double));
  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> elapsed_seconds;


  //-------[Sequential]--------
  start = std::chrono::system_clock::now();
  while(iter<maxIter){
    Jacobi<RAJA::seq_exec>(I,Iold);
    //Jacobi(I,Iold);
    iter++;
  }
  end = std::chrono::system_clock::now();

  elapsed_seconds = end-start;
  std::cout << "Seq elapsed time: " << elapsed_seconds.count() << "s\n";

  //Reset data
  iter = 0; 
  memset(I, 0, NN * sizeof(double));
  memset(Iold, 0, NN * sizeof(double));


#if 0
  //-------[Loop]--------
  start = std::chrono::system_clock::now();
  while(iter<maxIter){
    Jacobi<RAJA::loop_exec>(I,Iold);
    iter++;
  }
  end = std::chrono::system_clock::now();

  elapsed_seconds = end-start;
  std::cout << "loop elapsed time: " << elapsed_seconds.count() << "s\n";

  //Reset data
  iter = 0; 
  memset(I, 0, NN * sizeof(double));
  memset(Iold, 0, NN * sizeof(double));
#endif


  //-------[SIMD]---------
  start = std::chrono::system_clock::now();
  while(iter<maxIter){
    Jacobi<RAJA::simd_exec>(I,Iold);
    iter++;
  }
  end = std::chrono::system_clock::now();

  elapsed_seconds = end-start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "simd elapsed time: " << elapsed_seconds.count() << "s\n";


  //free(I);
  //free(Iold);
  return 0;
}
