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

#define CUDA_BLOCK_SIZE 256

void checkSolution(double *C, int in_N);

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  std::cout<<"Example 2: Multiplying Two Matrices"<<std::endl;

  const int N  = 1000;
  const int NN = N*N;
  double *A = new double[NN];
  double *B = new double[NN];
  double *C = new double[NN];

  //Populate Matrix
  for(int i=0; i<NN; ++i){
    A[i] = 1.0;
    B[i] = 1.0;
  }


  //----[Standard C++ loop]---------------
  std::cout<<"Standard C++ loop"<<std::endl;
  for(int i=0; i<NN; ++i){

    int c = i%N; int r=i/N;

    int cId = c+r*N; double dot=0.0;
    for(int k=0; k<N; ++k){
      int aId = k + r*N;
      int bId = c + k*N;
      dot += A[aId]*B[bId];
    }
    
    C[cId] = dot;
  }
  checkSolution(C,N);
  std::cout<<"\n"<<std::endl;
  //======================================


  //----[RAJA: Sequential Policy]---------
  std::cout<<"RAJA: Sequential Policy"<<std::endl;
  RAJA::forall<RAJA::seq_exec>(0,NN,[=](int i){

      int c = i%N; int r=i/N;
      
      int cId = c+r*N; double dot=0.0;
      for(int k=0; k<N; ++k){
        int aId = k + r*N;
        int bId = c + k*N;
        dot += A[aId]*B[bId];
      }    
      C[cId] = dot;
    });
  checkSolution(C,N);
  std::cout<<"\n"<<std::endl;
  //======================================


#if defined(RAJA_ENABLE_OPENMP)
  //----[RAJA: OpenMP Policy]---------
  std::cout<<"RAJA: OpenMP Policy"<<std::endl;
  RAJA::forall<RAJA::omp_parallel_for_exec>(0,NN,[=](int i){ 

      int c = i%N; int r=i/N;      
      int cId = c+r*N; double dot=0.0;
      for(int k=0; k<N; ++k){
        int aId = k + r*N;
        int bId = c + k*N;
        dot += A[aId]*B[bId];
      }    
      C[cId] = dot;

    });
  checkSolution(C,N);
  std::cout<<"\n"<<std::endl;
  //======================================
#endif



#if defined(RAJA_ENABLE_CUDA)
  //----[RAJA: CUDA Policy]---------
  std::cout<<"RAJA: CUDA Policy"<<std::endl;

  double *d_A, *d_B, *d_C;

  //User must manage host-device transfer
  cudaMallocManaged((void**)&d_A,sizeof(double)*NN,cudaMemAttachGlobal);
  cudaMallocManaged((void**)&d_B,sizeof(double)*NN,cudaMemAttachGlobal);
  cudaMallocManaged((void**)&d_C,sizeof(double)*NN,cudaMemAttachGlobal);

  //Copy data from host
  cudaMemcpy(d_A,A,sizeof(double)*NN,cudaMemcpyHostToDevice);
  cudaMemcpy(d_B,B,sizeof(double)*NN,cudaMemcpyHostToDevice);

  RAJA::forall<RAJA::cuda_exec<CUDA_BLOCK_SIZE>>(0,NN,[=] __device__(int i){

      int c = i%N; int r=i/N;      
      int cId = c+r*N; double dot=0.0;
      for(int k=0; k<N; ++k){
        int aId = k + r*N;
        int bId = c + k*N;
        dot += d_A[aId]*d_B[bId];
      }    
      d_C[cId] = dot;
    });

  checkSolution(d_C,N);
  std::cout<<"\n"<<std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  //======================================
#endif



  delete[] A, B, C;

  
  return 0;
}

void checkSolution(double *C, int in_N){

  bool eFlag = true;
  for(int id=0; id<in_N*in_N; ++id){
    if( abs(C[id]-in_N) > 1e-9){
      std::cout<<"Error in Result!"<<std::endl;
      eFlag = false;
      break;
    }
  }
  
  if(eFlag){
    std::cout<<"Result is correct"<<std::endl;
  }
}
