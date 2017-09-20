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
#include <cstring>

#include <iostream>
#include <cmath>

#include "RAJA/RAJA.hpp"
#include "RAJA/util/defines.hpp"

#include "memoryManager.hpp"

/*
  Example 3: Jacobi Method

  ----[Details]--------------------
  This code uses a five point finite difference stencil
  to discretize the following boundary value problem

  U_xx + U_yy = f on [0,1] x [0,1].

  The right-hand side is chosen to be
  f = 2*x*(y-1)*(y-2*x+x*y+2)*exp(x-y).

  A structured grid is used to discretize the domain
  [0,1] x [0,1]. Values inside the domain are computed
  using the Jacobi method to solve the associated
  linear system. The scheme is invoked until the l_2
  difference of subsequent iterations is below a
  tolerance.

  The scheme is implemented by allocating two arrays
  (I, Iold) and initialized to zero. The first set of
  nested for loops apply an iteration of the Jacobi
  scheme. As boundary values are already known the
  scheme is only applied to the interior nodes.

  The second set of nested for loops is used to
  update Iold and compute the l_2 norm of the
  difference of the iterates.

  Computing the l_2 norm requires a reduction operation.
  To simplify the reduction procedure, the RAJA API
  introduces thread safe variables.

  ----[RAJA Concepts]---------------
  1. ForallN loop
  2. RAJA Reduction
  3. RAJA::omp_collapse_nowait_exec

  ----[Kernel Variants and RAJA Features]---
  a. C++ style nested for loops
  b. RAJA style nested for loops with sequential iterations
     i. Introduces RAJA reducers for sequential policies
  c. RAJA style nested for loops with omp parallelism
     i.  Introduces collapsing loops using RAJA omp policies
     ii. Introduces RAJA reducers for omp policies
  d. RAJA style for loop with CUDA parallelism
     i. Introduces RAJA reducers for cuda policies
*/


/*
  ----[Constant Values]-----
  CUDA_BLOCK_SIZE_X - Number of threads in the
                      x-dimension of a cuda thread block

  CUDA_BLOCK_SIZE_Y - Number of threads in the
                      y-dimension of a cuda thread block

  CUDA_BLOCK_SIZE   - Number of threads per threads block
*/
const int CUDA_BLOCK_DIM_X = 16;
const int CUDA_BLOCK_DIM_Y = 16;
const int CUDA_BLOCK_SIZE = 256;


/*
  Struct to hold grid info
  o - Origin in a cartesian dimension
  h - Spacing between grid points
  n - Number of grid points
 */
struct grid_s {
  double o, h;
  int n;
};

/*
  ----[Functions]---------
  solution   - Function for the analytic solution
  computeErr - Displays the maximum error in the solution
*/
double solution(double x, double y);
void computeErr(double *I, grid_s grid);

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  printf("Example 3: Jacobi Method \n");

  /*
    ----[Solver Parameters]------------
    tol       - Method terminates once the norm is less than tol
    N         - Number of unknown gridpoints per cartesian dimension
    NN        - Total number of gridpoints on the grid
    maxIter   - Maximum number of iterations to be taken

    resI2     - Residual
    iteration - Iteration number
    grid_s    - Struct with grid information for a cartesian dimension
  */
  double tol = 1e-10;

  int N = 50;
  int NN = (N + 2) * (N + 2);
  int maxIter = 100000;

  double resI2;
  int iteration;

  grid_s gridx;
  gridx.o = 0.0;
  gridx.h = 1.0 / (N + 1.0);
  gridx.n = N + 2;

  /*
    I, Iold - Holds iterates of Jacobi method
  */
  double *I = memoryManager::allocate<double>(NN);
  double *Iold = memoryManager::allocate<double>(NN);


  memset(I, 0, NN * sizeof(double));
  memset(Iold, 0, NN * sizeof(double));


  printf("Standard  C++ Loop \n");
  resI2 = 1;
  iteration = 0;

  while (resI2 > tol * tol) {

    /*
      Jacobi Iteration
    */
    for (int n = 1; n <= N; ++n) {
      for (int m = 1; m <= N; ++m) {

        double x = gridx.o + m * gridx.h;
        double y = gridx.o + n * gridx.h;

        double f = gridx.h * gridx.h
                   * (2 * x * (y - 1) * (y - 2 * x + x * y + 2) * exp(x - y));

        int id = n * (N + 2) + m;
        I[id] = -0.25 * (f - Iold[id - N - 2] - Iold[id + N + 2] - Iold[id - 1]
                         - Iold[id + 1]);
      }
    }

    /*
      Compute residual and update Iold
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
  computeErr(I, gridx);
  printf("No of iterations: %d \n \n", iteration);


  /*
    RAJA loop calls may be shortened by predefining policies
  */
  RAJA::RangeSegment gridRange(0, NN);
  RAJA::RangeSegment jacobiRange(1, (N + 1));
  using jacobiSeqNestedPolicy =
      RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec, RAJA::seq_exec>>;

  printf("RAJA: Sequential Policy - Nested ForallN \n");
  resI2 = 1;
  iteration = 0;
  memset(I, 0, NN * sizeof(double));
  memset(Iold, 0, NN * sizeof(double));

  while (resI2 > tol * tol) {

    /*
      Jacobi Iteration
    */
    RAJA::forallN<jacobiSeqNestedPolicy>(
      jacobiRange, jacobiRange, [=](RAJA::Index_type m, RAJA::Index_type n) {      

          double x = gridx.o + m * gridx.h;
          double y = gridx.o + n * gridx.h;

          double f = gridx.h * gridx.h
                     * (2 * x * (y - 1) * (y - 2 * x + x * y + 2) * exp(x - y));

          int id = n * (N + 2) + m;
          I[id] =
              -0.25 * (f - Iold[id - N - 2] - Iold[id + N + 2] - Iold[id - 1]
                       - Iold[id + 1]);
        });

    /*
      ----[Reduction step]---------
      The RAJA API introduces a thread-safe accumulation variable
      "ReduceSum" in order to perform reductions
    */
    RAJA::ReduceSum<RAJA::seq_reduce, double> RAJA_resI2(0.0);
    RAJA::forall<RAJA::seq_exec>(
      gridRange, [=](RAJA::Index_type k) {
      
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
  computeErr(I, gridx);
  printf("No of iterations: %d \n \n", iteration);
  
  
#if defined(RAJA_ENABLE_OPENMP)
  printf("RAJA: OpenMP Policy - Nested ForallN \n");
  resI2 = 1;
  iteration = 0;
  memset(I, 0, NN * sizeof(double));
  memset(Iold, 0, NN * sizeof(double));
  
  /*
    ----[RAJA Policies]-----------
    RAJA::omp_collapse_nowait_exec -
    parallizes nested loops without introducing nested parallism

    RAJA::OMP_Parallel<> - Creates a parallel region,
    must be the last argument of the nested policy list
  */
  using jacobiompNestedPolicy =
    RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_collapse_nowait_exec,
    RAJA::omp_collapse_nowait_exec>, RAJA::OMP_Parallel<>>;

  while (resI2 > tol * tol) {

    /*
      Jacobi Iteration
    */
    RAJA::forallN<jacobiompNestedPolicy>(
        jacobiRange, jacobiRange, [=](RAJA::Index_type m, RAJA::Index_type n) {
                
          double x = gridx.o + m * gridx.h;
          double y = gridx.o + n * gridx.h;

          double f = gridx.h * gridx.h
                     * (2 * x * (y - 1) * (y - 2 * x + x * y + 2) * exp(x - y));

          int id = n * (N + 2) + m;
          I[id] = -0.25 * (f - Iold[id - N - 2] - Iold[id + N + 2] - Iold[id - 1]
                             - Iold[id + 1]);              
        });
    /*
      Compute residual and update Iold
    */
    RAJA::ReduceSum<RAJA::omp_reduce, double> RAJA_resI2(0.0);
    RAJA::forall<RAJA::omp_parallel_for_exec>(
      gridRange, [=](RAJA::Index_type k) {
      
        RAJA_resI2 += (I[k] - Iold[k]) * (I[k] - Iold[k]);                    
        Iold[k] = I[k];
        
      });
    
    resI2 = RAJA_resI2;
    if (iteration > maxIter) {
      printf("RAJA: OpenMP - Maxed out on iterations! \n");
      exit(-1);
    }
    iteration++;
  }
  computeErr(I, gridx);
  printf("No of iterations: %d \n \n", iteration);
#endif


#if defined(RAJA_ENABLE_CUDA)
  printf("RAJA: CUDA Policy - Nested ForallN \n");

  using jacobiCUDANestedPolicy = RAJA::NestedPolicy<RAJA::    
    ExecList<RAJA::cuda_threadblock_y_exec<CUDA_BLOCK_DIM_X>,
    RAJA::cuda_threadblock_x_exec<CUDA_BLOCK_DIM_Y>>>;    

  resI2 = 1;
  iteration = 0;
  memset(I, 0, NN * sizeof(double));
  memset(Iold, 0, NN * sizeof(double));

  while (resI2 > tol * tol) {

    /*
      Jacobi Iteration
    */
    RAJA::forallN<jacobiCUDANestedPolicy>(
        jacobiRange, jacobiRange, [=] __device__(RAJA::Index_type m, RAJA::Index_type n) {
        
          double x = gridx.o + m * gridx.h;
          double y = gridx.o + n * gridx.h;

          double f = gridx.h * gridx.h
                     * (2 * x * (y - 1) * (y - 2 * x + x * y + 2) * exp(x - y));

          int id = n * (N + 2) + m;
          I[id] = -0.25 * (f - Iold[id - N - 2] - Iold[id + N + 2] - Iold[id - 1]
                             - Iold[id + 1]);                            
        });

    /*
      Compute residual and update Iold
    */
    RAJA::ReduceSum<RAJA::cuda_reduce<CUDA_BLOCK_SIZE>, double> RAJA_resI2(0.0);
    RAJA::forall<RAJA::cuda_exec<CUDA_BLOCK_SIZE>>(
      gridRange, [=] __device__(RAJA::Index_type k) {
      
          RAJA_resI2 += (I[k] - Iold[k]) * (I[k] - Iold[k]);
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
  computeErr(I, gridx);
  printf("No of iterations: %d \n \n", iteration);
#endif

  memoryManager::deallocate(I);
  memoryManager::deallocate(Iold);
  

  return 0;
}

/*
  Function for the anlytic solution
*/
double solution(double x, double y)
{
  return x * y * exp(x - y) * (1 - x) * (1 - y);
}

/*
  Error is computed via ||I_{approx}(:) - U_{analytic}(:)||_{inf}
*/
void computeErr(double *I, grid_s grid)
{

  RAJA::RangeSegment fdBounds(0, grid.n);
  RAJA::ReduceMax<RAJA::seq_reduce, double> tMax(-1.0);
  using myPolicy =
    RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec, RAJA::seq_exec>>;

  RAJA::forallN<myPolicy>(
    fdBounds, fdBounds, [=](RAJA::Index_type ty, RAJA::Index_type tx) {    

      int id = tx + grid.n * ty;
      double x = grid.o + tx * grid.h;
      double y = grid.o + ty * grid.h;
      double myErr = std::abs(I[id] - solution(x, y));
      tMax.max(myErr);
    });

  double l2err = tMax;
  printf("Max error = %lg, h = %f \n", l2err, grid.h);
}
