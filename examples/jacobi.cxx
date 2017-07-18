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
#include <cstdio> 
#include <iostream>

#include "RAJA/RAJA.hpp"
#include "RAJA/util/defines.hpp"


#define block_size 256

// Solve for loop currents in structured resistor array.
// Similar structure to discretizing the poisson equation 
// with a second order finite difference scheme
int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  //----[Setting up solver]-------
  double tol = 1e-5;
  int maxIter = 10000;
  int N = 100;
  int NN=(N+2)*(N+2);
  unsigned int iteration;
  double resI2, V, invD;
  double *I    = new double [NN]; 
  double *Iold = new double [NN]; 

  memset(I,0,NN*sizeof(double));
  memset(Iold,0,NN*sizeof(double));

  //----[Standard C approach]-----
  resI2 = 1;  iteration = 0; 
  while(resI2>tol*tol){
    
    resI2 = 0;    
    for(unsigned int n=1;n<=N;++n){
      for(unsigned int m=1;m<=N;++m){

        unsigned int id = n*(N+2) + m;
        //Cell (1,1) is a special case
        if(n==1 && m==1){
          invD = 1./3.; V =1; 
          I[id] = invD*(V-Iold[id-N-2]-Iold[id+N+2]-Iold[id-1]-Iold[id+1]);
        }else{
          invD = 1./4.0; V = 0.0;    
          I[id] = invD*(V-Iold[id-N-2]-Iold[id+N+2]-Iold[id-1]-Iold[id+1]);
        }
        
      }
    }

    //Reduction step
    for(unsigned int k=0; k<NN; k++){
      resI2 += (I[k]-Iold[k])*(I[k]-Iold[k]);
      Iold[k]=I[k];
    }

    if(iteration > maxIter){        
      printf("Standard C - too many iterations!\n");
      exit(-1);
    }

    iteration++;
  }
  printf("Standard C/Cpp Loops: \n");
  printf("Top right current: %lg \n", I[N+N*(N+2)]);
  printf("No of iterations: %d \n \n",iteration);
  //======================================

  

  //----[RAJA: Nested Sequential Policy]---------
  //RAJA does not allow variables to be modified inside the loop
  resI2 = 1; iteration = 0; 
  memset(I,0,NN*sizeof(double));
  memset(Iold,0,NN*sizeof(double));
  while(resI2 > tol*tol){
    
    
    RAJA::forallN< RAJA::NestedPolicy<
    RAJA::ExecList<RAJA::seq_exec,RAJA::seq_exec >>>(
    RAJA::RangeSegment(1, (N+1)),
    RAJA::RangeSegment(1, (N+1)),
    [=](int m, int n) { 

      unsigned int id = n*(N+2) + m;
      if(n==1 && m==1){
        double invD = 1./3.; double V = 1; 
        I[id] = invD*(V-Iold[id-N-2]-Iold[id+N+2]-Iold[id-1]-Iold[id+1]);
      }else{
        double invD2 = 1./4.0; double V2 = 0.0;
        I[id] = invD2*(V2-Iold[id-N-2]-Iold[id+N+2]-Iold[id-1]-Iold[id+1]);
      }

  });
    
    //Reduction step    
    RAJA::ReduceSum<RAJA::seq_reduce, double> RAJA_resI2(0.0);
    RAJA::forall<RAJA::seq_reduce>(0, NN, [=](int k) {
        RAJA_resI2 += (I[k]-Iold[k])*(I[k]-Iold[k]);
        Iold[k]=I[k];
      });
    
    
    resI2 = RAJA_resI2; 
    if(iteration > maxIter){        
      printf("RAJA-Seq - too many iterations\n");
      exit(-1);
    }
    iteration++;
  }
  printf("RAJA Nested Loop Sequential Policy: \n");
  printf("Top right current: %lg \n", I[N+N*(N+2)]);
  printf("No of iterations: %d \n \n",iteration);
  //======================================


  //----[RAJA: Nested OpenMP Policy]---------
  //RAJA does not allow variables to be modified inside the loop
  resI2 = 1; iteration = 0; 
  memset(I,0,NN*sizeof(double));
  memset(Iold,0,NN*sizeof(double));
  while(resI2 > tol*tol){
    
    
    RAJA::forallN< RAJA::NestedPolicy<
    RAJA::ExecList<RAJA::omp_parallel_for_exec,RAJA::omp_parallel_for_exec>>>(
    RAJA::RangeSegment(1, (N+1)),
    RAJA::RangeSegment(1, (N+1)),
    [=](int m, int n) { 
      
      unsigned int id = n*(N+2) + m;
      if(n==1 && m==1){
        double invD = 1./3.; double V = 1; 
        I[id] = invD*(V-Iold[id-N-2]-Iold[id+N+2]-Iold[id-1]-Iold[id+1]);
      }else{
        double invD2 = 1./4.0; double V2 = 0.0;
        I[id] = invD2*(V2-Iold[id-N-2]-Iold[id+N+2]-Iold[id-1]-Iold[id+1]);
      }

  });
    
    //Reduction step    
    RAJA::ReduceSum<RAJA::omp_reduce, double> RAJA_resI2(0.0);
    RAJA::forall<RAJA::omp_reduce>(0, NN, [=](int k) {
        RAJA_resI2 += (I[k]-Iold[k])*(I[k]-Iold[k]);
        Iold[k]=I[k];
      });
    
    
    resI2 = RAJA_resI2; 
    if(iteration > maxIter){        
      std::cout<<"RAJA:OpenMP - too many iterations!"<<std::endl;
      exit(-1);
    }
    iteration++;
  }
  printf("RAJA Nested Loop Sequential Policy: \n");
  printf("Top right current: %lg \n", I[N+N*(N+2)]);
  printf("No of iterations: %d \n \n",iteration);
  //======================================



  //----[RAJA: Nested CUDA Policy]---------
  resI2 = 1; iteration = 0; 
  memset(I,0,NN*sizeof(double));
  memset(Iold,0,NN*sizeof(double));
  
  double *d_I, *d_Iold;
  cudaMallocManaged((void**)&d_I,sizeof(double)*NN,cudaMemAttachGlobal);
  cudaMallocManaged((void**)&d_Iold,sizeof(double)*NN,cudaMemAttachGlobal);
  
  //copy data from host
  cudaMemcpy(d_I,I,sizeof(double)*NN,cudaMemcpyHostToDevice);
  cudaMemcpy(d_Iold,Iold,sizeof(double)*NN,cudaMemcpyHostToDevice);
  
  while(resI2 > tol*tol){
    
    if(iteration==0){
      std::cout<<"Entered the for looop"<<std::endl;
    }
    
    RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::cuda_threadblock_y_exec<16>,
    RAJA::cuda_threadblock_x_exec<16>>>>(
    RAJA::RangeSegment(1, (N+1)), RAJA::RangeSegment(1, (N+1)), [=] __device__ (int m, int n) {
     
      unsigned int id = n*(N+2) + m;
      if(n==1 && m==1){
        double invD = 1./3.; double V = 1; 
        d_I[id] = invD*(V-d_Iold[id-N-2]-d_Iold[id+N+2]-d_Iold[id-1]-d_Iold[id+1]);
      }else{
        double invD2 = 1./4.0; double V2 = 0.0;
        d_I[id] = invD2*(V2-d_Iold[id-N-2]-d_Iold[id+N+2]-d_Iold[id-1]-d_Iold[id+1]);
      }
    });

    //Reduction step    
    RAJA::ReduceSum<RAJA::cuda_reduce<256>, double> RAJA_resI2(0.0);
    RAJA::forall<RAJA::cuda_exec<256> >(0, NN, [=] __device__ (int k) {
        RAJA_resI2 += (d_I[k]-d_Iold[k])*(d_I[k]-d_Iold[k]);
        d_Iold[k]=d_I[k];
      });

    resI2 = RAJA_resI2; 
    
    if(iteration > maxIter){        
      std::cout<<"CUDA : too many iterations!"<<std::endl;
      exit(-1);
    }
    iteration++;
  }
  printf("RAJA Nested Loop CUDA Policy: \n");
  printf("Top right current: %lg \n", d_I[N+N*(N+2)]);
  printf("No of iterations: %d \n \n",iteration);
  //======================================
  
  //Clean up
  delete [] I, Iold;

  return 0;
}
