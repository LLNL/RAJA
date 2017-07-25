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


#define CUDA_BLOCK_SIZE 16

template <typename T>
T* allocate(RAJA::Index_type size) {
  T* ptr;
#if defined(RAJA_ENABLE_CUDA)
  cudaMallocManaged((void**)&ptr, sizeof(T)*size,cudaMemAttachGlobal);
#else
  ptr = new T[size];
#endif
  return ptr;
}

template <typename T>
void deallocate(T* &ptr) {
  if (ptr) {
#if defined(RAJA_ENABLE_CUDA)
    cudaFree(ptr);
#else
    delete[] ptr;
#endif
    ptr = nullptr;
  }
}



/*Example 3: Jacobi Method For Solving a Linear System

  ----[Details]--------------------
  The following code uses the jacobi method to compute the currents in a
  structured grid by solving a linear system correspoding to kirchoff's 
  circuit law. 

  The structured N x N circuit may be visualized as follows: 

  - Dash lines correspond to lossless wires,
  - The circles correspond to lossless juntions
  - The [] correspond to 1 Ohm resistors
  - The [x] corresponds to a 1 volt DC battery which powers the grid
  
  o--[]--o--[]--o--[]--o--[]--o
  |      |      |      |      |  
  []    []     []     []     []     
  |      |      |      |      |
  o--[]--o--[]--o--[]--o--[]--o
  |      |      |      |      |  
  []    []     []     []     []     
  |      |      |      |      |
  o--[]--o--[]--o--[]--o--[]--o
  |      |      |      |      |  
  []    []     []     []     []     
  |      |      |      |      |
  o--[]--o--[]--o--[]--o--[]--o
  |      |      |      |      |  
  [x]    []     []     []     []     
  |      |      |      |      |
  o--[]--o--[]--o--[]--o--[]--o


  ----[RAJA Concepts]---------------
  1. RAJA Reduction
  2. RAJA::omp_collapse_nowait_exec
*/

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  std::cout<<"Example 3: Jacobi Method for Linear System"<<std::endl;

  //----[Parameters for Solver]-------
  double tol = 1e-5;
  int maxIter = 10000;
  int N = 100;
  int NN=(N+2)*(N+2);
  int iteration;
  double resI2, V, invD;

  double *I    = allocate<double>(NN);
  double *Iold = allocate<double>(NN);

  //Set to zero
  memset(I,0,NN*sizeof(double));
  memset(Iold,0,NN*sizeof(double));


  //----[Standard C++ Loops]-----
  printf("Traditional  C++ Loop \n");
  resI2 = 1;  iteration = 0; 

  //Carry out iterations of the Jacobi scheme until a tolerance is met
  while(resI2>tol*tol){
    
    //Iteration of the Jacobi Scheme
    for(int n=1;n<=N;++n){
      for(int m=1;m<=N;++m){

        int id = n*(N+2) + m;
        //Cell (1,1) is a special case due to battery
        if(n==1 && m==1){
          invD = 1./3.; V =1; 
          I[id] = invD*(V-Iold[id-N-2]-Iold[id+N+2]-Iold[id-1]-Iold[id+1]);
        }else{
          invD = 1./4.0; V = 0.0;    
          I[id] = invD*(V-Iold[id-N-2]-Iold[id+N+2]-Iold[id-1]-Iold[id+1]);
        }
        
      }
    }

    //Compute ||I - I_{old}||_{\ifinity}    
    resI2 = 0.0; 
    for(int k=0; k<NN; k++){
      resI2 += (I[k]-Iold[k])*(I[k]-Iold[k]);
      Iold[k]=I[k];
    }
    
    if(iteration > maxIter){        
      printf("Traditional C++ loop - Maxed out on iterations \n");
      exit(-1);
    }

    iteration++;
  }
  printf("Top right current: %lg \n", I[N+N*(N+2)]);
  printf("No of iterations: %d \n \n",iteration);
  //======================================

  //----[RAJA: Nested Sequential Policy]---------
  printf("RAJA: Sequential Nested Loop Policy \n"); 
  resI2 = 1; iteration = 0; 
  memset(I,0,NN*sizeof(double));
  memset(Iold,0,NN*sizeof(double));
  while(resI2 > tol*tol){
        
    RAJA::forallN< RAJA::NestedPolicy<
    RAJA::ExecList<RAJA::seq_exec,RAJA::seq_exec >>>(
    RAJA::RangeSegment(1, N+1),
    RAJA::RangeSegment(1, N+1),
    [=](int m, int n) { 

      int id = n*(N+2) + m;
      if(n==1 && m==1){
        double invD = 1./3.; double V = 1; 
        I[id] = invD*(V-Iold[id-N-2]-Iold[id+N+2]-Iold[id-1]-Iold[id+1]);
      }else{
        double invD2 = 1./4.0; double V2 = 0.0;
        I[id] = invD2*(V2-Iold[id-N-2]-Iold[id+N+2]-Iold[id-1]-Iold[id+1]);
      }

  });
    
    /*----[Reduction step]---------
      The RAJA API introduces an accumulation variable
      "ReduceSum" which creates a thread-safe variable tailored to
      each execution policy. 

      Analogous to introducing an atomic operator in OpenMP
    */
    RAJA::ReduceSum<RAJA::seq_reduce, double> RAJA_resI2(0.0);
    RAJA::forall<RAJA::seq_reduce>(0, NN, [=](int k) {
        RAJA_resI2 += (I[k]-Iold[k])*(I[k]-Iold[k]);
        Iold[k]=I[k];
      });
    
    
    resI2 = RAJA_resI2; 
    if(iteration > maxIter){        
      printf("RAJA::Sequential - Maxed out on iterations! \n");
      exit(-1);
    }
    iteration++;
  }
  printf("Top right current: %lg \n", I[N+N*(N+2)]);
  printf("No of iterations: %d \n \n",iteration);
  //======================================


#if defined(RAJA_ENABLE_OPENMP)
  //----[RAJA: Nested OpenMP Policy]---------
  resI2 = 1; iteration = 0; 
  memset(I,0,NN*sizeof(double));
  memset(Iold,0,NN*sizeof(double));
  while(resI2 > tol*tol){
        
    /*
      RAJA::omp_collapse_nowait_exec -
      parallizes nested loops without introducing nested parallism
    */

    RAJA::forallN< RAJA::NestedPolicy<
    RAJA::ExecList<RAJA::omp_collapse_nowait_exec,RAJA::omp_collapse_nowait_exec>>>(
    RAJA::RangeSegment(1, (N+1)),
    RAJA::RangeSegment(1, (N+1)),
    [=](int m, int n) { 
      
      int id = n*(N+2) + m;
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
      printf("RAJA::OpenMP - Maxed out on iterations! \n");
      exit(-1);
    }
    iteration++;
  }
  printf("RAJA: OpenMP Nested Loop Policy \n"); 
  printf("Top right current: %lg \n", I[N+N*(N+2)]);
  printf("No of iterations: %d \n \n",iteration);
  //======================================
#endif


#if defined(RAJA_ENABLE_CUDA)
  //----[RAJA: Nested CUDA Policy]---------
  resI2 = 1; iteration = 0; 
  memset(I,0,NN*sizeof(double));
  memset(Iold,0,NN*sizeof(double));
    
  while(resI2 > tol*tol){
        
    RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::cuda_threadblock_y_exec<CUDA_BLOCK_SIZE>,
    RAJA::cuda_threadblock_x_exec<CUDA_BLOCK_SIZE>>>>(
    RAJA::RangeSegment(1, (N+1)), RAJA::RangeSegment(1, (N+1)), [=] __device__ (int m, int n) {
     
      int id = n*(N+2) + m;
      if(n==1 && m==1){
        double invD = 1./3.; double V = 1; 
        I[id] = invD*(V-Iold[id-N-2]-Iold[id+N+2]-Iold[id-1]-Iold[id+1]);
      }else{
        double invD2 = 1./4.0; double V2 = 0.0;
        I[id] = invD2*(V2-Iold[id-N-2]-Iold[id+N+2]-Iold[id-1]-Iold[id+1]);
      }
    });

    //Reduction step 
    RAJA::ReduceSum<RAJA::cuda_reduce<256>, double> RAJA_resI2(0.0);
    RAJA::forall<RAJA::cuda_exec<256> >(0, NN, [=] __device__ (int k) {
        RAJA_resI2 += (I[k]-Iold[k])*(I[k]-Iold[k]);
        Iold[k]=I[k];
      });

    resI2 = RAJA_resI2; 
    
    if(iteration > maxIter){        
      std::cout<<"CUDA : too many iterations!"<<std::endl;
      exit(-1);
    }
    iteration++;
  }
  cudaDeviceSynchronize();
  printf("RAJA: CUDA Nested Loop Policy \n"); 
  printf("Top right current: %lg \n", I[N+N*(N+2)]);
  printf("No of iterations: %d \n \n",iteration);
  //======================================
#endif

  deallocate(I);
  deallocate(Iold);

  return 0;
}
