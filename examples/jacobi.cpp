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
#define CUDA_REDUCE_SIZE 256


/*
  Example 3: Jacobi Method

  ----[Details]--------------------
  This codes uses a five point finite difference stencil
  to discretize 
  
  -U_xx - U_yy = 1.0.
  The solution is assumed to satisfy
  U = 0 on the boundary of the domain.

  Values inside the domain are computed
  using the Jocabi method to solve the associated
  linear system. The domain is assumed to be a lattice 
  with unit grid spacing.


  ----[RAJA Concepts]---------------
  1. RAJA ForallN
  2. RAJA Reduction
  3. RAJA::omp_collapse_nowait_exec
  3. Reducing length of RAJA statements
*/

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  printf("Example 3: Jacobi Method \n");

  // Jacobi method is invoked until the l_2 norm of the difference of the
  // iterations are under a tolerance
  double tol = 1e-5;

  int N       = 100;                // Number of unknown gridpoints in a cartesian dimension
  int NN      = (N + 2) * (N + 2);  // Total number of gridpoints on the lattice
  int maxIter = 100000;             // Maximum number of iterations to be taken

  double resI2;                     // Residual
  int iteration;                    // Iteration number

  /*
    f       - Right hand side term
    invD    - Accumulation of finite difference coefficients
    I, Iold - Holds iterates of Jacobi method
  */
  double f     = 1.0;
  double invD  = 1. / 4.0;
  double *I    = memoryManager::allocate<double>(NN);
  double *Iold = memoryManager::allocate<double>(NN);

  memset(I, 0, NN * sizeof(double));
  memset(Iold, 0, NN * sizeof(double));


  printf("Standard  C++ Loop \n");
  resI2 = 1;
  iteration = 0;

  while (resI2 > tol * tol) {

    for (int n = 1; n <= N; ++n) {
      for (int m = 1; m <= N; ++m) {

        int id = n * (N + 2) + m;
        I[id] = invD * (f - Iold[id - N - 2] - Iold[id + N + 2] 
                          - Iold[id - 1] - Iold[id + 1]);        
      }
    }

    /*
      Residual is computed via ||I - I_{old}||_{l2}
      I_{old} is updated
    */
    resI2 = 0.0;
    for (int k = 0; k < NN; k++) {
      resI2 += (I[k] - Iold[k]) * (I[k] - Iold[k]);
      Iold[k] = I[k];
    }
    
    if (iteration > maxIter) {
      printf("Standard C++ Loop - Maxed out on iterations \n");
      exit(-1);
    }
    
    iteration++;
  }
  printf("Value at grid point ((N-1), (N-1)): %lg \n", I[N + N * (N + 2)]);
  printf("No of iterations: %d \n \n", iteration);
  
  /*
    RAJA loop calls may be shortened by defining policies before hand
  */
  RAJA::RangeSegment gridRange(0, NN);
  RAJA::RangeSegment jacobiRange(1, (N + 1));
  using jacobiSeqNestedPolicy =
    RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec, RAJA::seq_exec>>;

  printf("RAJA: Sequential Policy - Nested forallN \n");
  resI2 = 1;
  iteration = 0;
  memset(I, 0, NN * sizeof(double));
  memset(Iold, 0, NN * sizeof(double));

  while (resI2 > tol * tol) {

    RAJA::forallN<jacobiSeqNestedPolicy>
      (jacobiRange, jacobiRange, [=](RAJA::Index_type m,RAJA::Index_type n) {
        
        int id = n * (N + 2) + m;
        I[id] = invD * (f - Iold[id - N - 2] - Iold[id + N + 2]
                        - Iold[id - 1] - Iold[id + 1]);
      });

    /*
      ----[Reduction step]---------
      The RAJA API introduces a thread-safe accumulation variable
      "ReduceSum" in order to carryout reductions.
    */
    RAJA::ReduceSum<RAJA::seq_reduce, double> RAJA_resI2(0.0);
    RAJA::forall<RAJA::seq_exec>
      (gridRange, [=](RAJA::Index_type k) {
        
        RAJA_resI2 += (I[k] - Iold[k]) * (I[k] - Iold[k]);
        Iold[k] = I[k];
      });

    resI2 = RAJA_resI2;
    if (iteration > maxIter) {
      printf("RAJA: Sequential - Maxed out on iterations! \n");
      exit(-1);
    }
    iteration++;
  }
  printf("Value at grid point ((N-1),(N-1)): %lg \n", I[N + N * (N + 2)]);
  printf("No of iterations: %d \n \n", iteration);
  
  
#if defined(RAJA_ENABLE_OPENMP)
  printf("RAJA: OpenMP Policy - Nested forallN \n");
  resI2 = 1;
  iteration = 0;
  memset(I, 0, NN * sizeof(double));
  memset(Iold, 0, NN * sizeof(double));

  /*
    RAJA::omp_collapse_nowait_exec -
    parallizes nested loops without introducing nested parallism

    RAJA::OMP_Parallel<> - Creates a parallel region, 
    must be the last argument of the nested policy list
  */
  using jacobiompNestedPolicy =
    RAJA::NestedPolicy<
    RAJA::ExecList<RAJA::omp_collapse_nowait_exec,
    RAJA::omp_collapse_nowait_exec>,
    RAJA::OMP_Parallel<>>;

  while (resI2 > tol * tol) {

    RAJA::forallN<jacobiompNestedPolicy>
      (jacobiRange, jacobiRange, [=](RAJA::Index_type m,RAJA::Index_type n) {
        
        int id = n * (N + 2) + m;
        I[id] = invD * (f - Iold[id - N - 2] - Iold[id + N + 2] - Iold[id - 1]
                        - Iold[id + 1]);
      });
    
    // Reduction step
    RAJA::ReduceSum<RAJA::omp_reduce, double> RAJA_resI2(0.0);    
    RAJA::forall<RAJA::omp_parallel_for_exec>
      (gridRange, [=](RAJA::Index_type k) {
        
        RAJA_resI2 += (I[k] - Iold[k])*(I[k] - Iold[k]);
        Iold[k] = I[k];
      });
    

    resI2 = RAJA_resI2;
    if (iteration > maxIter) {
      printf("RAJA: OpenMP - Maxed out on iterations! \n");
      exit(-1);
    }
    iteration++;
  }
  printf("Value at grid point ((N-1),(N-1)): %lg \n", I[N + N * (N + 2)]);
  printf("No of iterations: %d \n \n", iteration);
#endif

  
#if defined(RAJA_ENABLE_CUDA)
  using jacobiCUDANestedPolicy = 
    RAJA::NestedPolicy<RAJA::ExecList
    <RAJA::cuda_threadblock_y_exec<CUDA_BLOCK_SIZE>,
    RAJA::cuda_threadblock_x_exec<CUDA_BLOCK_SIZE>>>;
    
  resI2 = 1;
  iteration = 0;
  memset(I, 0, NN * sizeof(double));
  memset(Iold, 0, NN * sizeof(double));
  
  while (resI2 > tol * tol) {
    
    RAJA::forallN<jacobiCUDANestedPolicy>
      (jacobiRange, jacobiRange, [=] __device__(RAJA::Index_type m,RAJA::Index_type n) {
        
        int id = n * (N + 2) + m;
        I[id] = invD * (f - Iold[id - N - 2] - Iold[id + N + 2]
                        - Iold[id - 1] - Iold[id + 1]);
      });
    
    // Reduction step
    RAJA::ReduceSum<RAJA::cuda_reduce<CUDA_REDUCE_SIZE>, double> RAJA_resI2(0.0);
    RAJA::forall<RAJA::cuda_exec<CUDA_REDUCE_SIZE>>
      (gridRange, [=] __device__(RAJA::Index_type k) {
       
        RAJA_resI2 += (I[k] - Iold[k])*(I[k] - Iold[k]);          
        Iold[k] = I[k];
      });

    resI2 = RAJA_resI2;
    
    if (iteration > maxIter) {
      printf("RAJA: CUDA - Maxed out on iterations! \n");
      exit(-1);
    }
    iteration++;
  }
  cudaDeviceSynchronize();
  printf("RAJA: CUDA Nested Loop Policy \n");
  printf("Value at grid point ((N-1),(N-1)): %lg \n", I[N + N * (N + 2)]);
  printf("No of iterations: %d \n \n", iteration);
#endif

  memoryManager::deallocate(I);
  memoryManager::deallocate(Iold);


  return 0;
}
