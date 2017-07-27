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

#include "memoryManager.hpp"

#define CUDA_BLOCK_SIZE 16


/*
  Example 3: Jacobi Method For Solving a Linear System

  ----[Details]--------------------
  This code assumes a second order finite difference 
  spatial discretization on a lattice with unit grid spacing
  
  More specifically it solves:
  -U_xx - U_yy = 1.0 inside the domain 
  and U = 0 on the boundary of the domain

  ----[RAJA Concepts]---------------
  1. RAJA ForallN
  2. RAJA Reduction
  3. RAJA::omp_collapse_nowait_exec
  3. Reducing length of RAJA statements
*/

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  std::cout<<"Example 3: Jacobi Method for Linear System"<<std::endl;

  //----[Parameters for Solver]-------
  double tol = 1e-5;
  int maxIter = 100000;
  int N = 100;
  int NN=(N+2)*(N+2);
  int iteration;
  double resI2;

  /*
    invD - accumulation of finite difference coefficients 
    f - right hand side term          
  */
  double invD = 1./4.0;  double f = 1.0;    

  /*
    Variables to hold approximation
   */
  double *I    = memoryManager::allocate<double>(NN);
  double *Iold = memoryManager::allocate<double>(NN);

  /*
    Intialize data to zero
  */
  memset(I,0,NN*sizeof(double));
  memset(Iold,0,NN*sizeof(double));

  printf("Standard  C++ Loop \n");
  resI2 = 1;  iteration = 0; 

  /*
    Carry out iterations of the Jacobi scheme until a tolerance is met
  */
  while(resI2>tol*tol) {
    
    /*
    Iteration of the Jacobi Scheme
    */
    for(int n=1;n<=N;++n) {
      for(int m=1;m<=N;++m) {

        int id = n*(N+2) + m;
                
        I[id] = invD*(f-Iold[id-N-2]-Iold[id+N+2]-Iold[id-1]-Iold[id+1]);                
      }
    }

    /*
      Residual is computed via ||I - I_{old}||_{l2} + furthermore I_{old} is updated
    */
    resI2 = 0.0; 
    for(int k=0; k<NN; k++) {
      resI2 += (I[k]-Iold[k])*(I[k]-Iold[k]);
      Iold[k]=I[k];
    }
    
    if(iteration > maxIter) {        
      printf("Standard C++ Loop - Maxed out on iterations \n");
      exit(-1);
    }

    iteration++;
  }
  printf("Value at grid point ((N-1), (N-1)): %lg \n", I[N+N*(N+2)]);
  printf("No of iterations: %d \n \n",iteration);
  //======================================
  
  /*
    RAJA loop calls may be shortened by defining policies before hand
  */
  RAJA::RangeSegment jacobiRange = RAJA::RangeSegment(1,(N+1)); 
  RAJA::RangeSegment gridRange = RAJA::RangeSegment(0,NN); 
  using jacobiSeqNestedPolicy = RAJA::NestedPolicy< RAJA::ExecList< RAJA::seq_exec, RAJA::seq_exec > >;
  
  printf("RAJA: Sequential Policy - Nested forallN \n"); 
  resI2 = 1; iteration = 0; 
  memset(I,0,NN*sizeof(double));
  memset(Iold,0,NN*sizeof(double));

  while(resI2 > tol*tol) {

    RAJA::forallN< jacobiSeqNestedPolicy >(jacobiRange,jacobiRange, [=](int m, int n) { 
      int id = n*(N+2) + m;      
      I[id] = invD*(f-Iold[id-N-2]-Iold[id+N+2]-Iold[id-1]-Iold[id+1]);
    });

    /*
      ----[Reduction step]---------
      The RAJA API introduces an accumulation variable
      "ReduceSum" which creates a thread-safe variable tailored to
      each execution policy. 

      Analogous to introducing an atomic operator in OpenMP
    */
    RAJA::ReduceSum<RAJA::seq_reduce, double> RAJA_resI2(0.0);
    RAJA::forall<RAJA::seq_exec>(gridRange, [=](int k) {
        RAJA_resI2 += (I[k]-Iold[k])*(I[k]-Iold[k]);
        Iold[k]=I[k];
      });
       
    resI2 = RAJA_resI2; 
    if(iteration > maxIter) {
      printf("RAJA: Sequential - Maxed out on iterations! \n");
      exit(-1);
    }
    iteration++;
  }
  printf("Value at grid point ((N-1),(N-1)): %lg \n", I[N+N*(N+2)]);
  printf("No of iterations: %d \n \n",iteration);



#if defined(RAJA_ENABLE_OPENMP)
  printf("RAJA: OpenMP Policy - Nested forallN \n");
  resI2 = 1; iteration = 0; 
  memset(I,0,NN*sizeof(double));
  memset(Iold,0,NN*sizeof(double));
  
  /*
    RAJA::omp_collapse_nowait_exec -
    parallizes nested loops without introducing nested parallism
    
    RAJA::OMP_Parallel<> - Creates a parallel region
  */
  using jacobiompNestedPolicy = 
    RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_collapse_nowait_exec,RAJA::omp_collapse_nowait_exec>,RAJA::OMP_Parallel<>>;

  while(resI2 > tol*tol) {
            
    RAJA::forallN<jacobiompNestedPolicy>(jacobiRange,jacobiRange,[=](int m, int n) { 
                                                                    
      int id = n*(N+2) + m;      
      I[id] = invD*(f-Iold[id-N-2]-Iold[id+N+2]-Iold[id-1]-Iold[id+1]);
  });
    
    //Reduction step    
    RAJA::ReduceSum<RAJA::omp_reduce, double> RAJA_resI2(0.0);
    RAJA::forall<RAJA::omp_parallel_for_exec>(gridRange, [=](int k){
        RAJA_resI2 += (I[k]-Iold[k])*(I[k]-Iold[k]);
        Iold[k]=I[k];
      });
    
    
    resI2 = RAJA_resI2; 
    if(iteration > maxIter) {
      printf("RAJA: OpenMP - Maxed out on iterations! \n");
      exit(-1);
    }
    iteration++;
  }
  printf("Value at grid point ((N-1),(N-1)): %lg \n", I[N+N*(N+2)]);
  printf("No of iterations: %d \n \n",iteration);

#endif


#if defined(RAJA_ENABLE_CUDA)
  using jacobiCUDANestedPolicy = 
    RAJA::NestedPolicy<RAJA::ExecList<RAJA::cuda_threadblock_y_exec<CUDA_BLOCK_SIZE>,RAJA::cuda_threadblock_x_exec<CUDA_BLOCK_SIZE>>>;       

  resI2 = 1; iteration = 0; 
  memset(I,0,NN*sizeof(double));
  memset(Iold,0,NN*sizeof(double));
    
  while(resI2 > tol*tol) {
        
    RAJA::forallN<jacobiCUDANestedPolicy>(jacobiRange,jacobiRange, [=] __device__ (int m, int n) {
      int id = n*(N+2) + m;      
      I[id] = invD*(f-Iold[id-N-2]-Iold[id+N+2]-Iold[id-1]-Iold[id+1]);     

    });

    //Reduction step 
    RAJA::ReduceSum<RAJA::cuda_reduce<256>, double> RAJA_resI2(0.0);
    RAJA::forall<RAJA::cuda_exec<256> >(gridRange, [=] __device__ (int k) {
        RAJA_resI2 += (I[k]-Iold[k])*(I[k]-Iold[k]);
        Iold[k]=I[k];
      });

    resI2 = RAJA_resI2; 
    
    if(iteration > maxIter) {
      printf("RAJA: CUDA - Maxed out on iterations! \n");
      exit(-1);
    }
    iteration++;
  }
  cudaDeviceSynchronize();
  printf("RAJA: CUDA Nested Loop Policy \n"); 
  printf("Value at grid point ((N-1),(N-1)): %lg \n", I[N+N*(N+2)]);
  printf("No of iterations: %d \n \n",iteration);
  //======================================
#endif

  memoryManager::deallocate(I);
  memoryManager::deallocate(Iold);


  return 0;
}
