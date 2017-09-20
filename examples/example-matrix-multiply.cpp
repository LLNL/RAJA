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
#include <cmath>
#include <iostream>
#include <algorithm>
#include <initializer_list>

#include "RAJA/RAJA.hpp"
#include "RAJA/index/RangeSegment.hpp"
#include "RAJA/util/defines.hpp"

#include "memoryManager.hpp"

/*
  Example 2: Multiplying Two Matrices

  ----[Details]--------------------
  Starting with C++ style nested for loops, this example
  illustrates how to construct RAJA versions of the same loops
  using different execution policies. Furthermore, as nesting
  RAJA forall loops are not currently supported with CUDA,
  this example makes utility of RAJA's forallN loop which
  may be used with any policy.

  In this example two matrices of dimension N x N are allocated and multiplied.
  The matrix A is populated with a constant value along the rows while B is
  populated with a constant value along the columns. The function checkSolution 
  checks for correctness.

  -----[RAJA Concepts]-------------
  1. Nesting forall loops (Not currently supported in CUDA)

  2. ForallN loop (Supported with all policies)

  RAJA::forallN<
  RAJA::NestedPolicy<exec_policy1, .... , exec_policyN> >(
  iter_space I1,..., iter_space IN, [=](index_type i1,..., index_type iN) {

         //body

  });

  [=] By-copy capture
  [&] By-reference capture (for non-unified memory targets)
  RAJA::NestedPolicy - Stores a list of RAJA execution policies
  exec_policy        - Specifies how the traversal occurs
  iter_space         - Iteration space for RAJA loop (any random access
  container is expected)

  3. RAJA::View - RAJA's wrapper for multidimensional indexing

  ----[Kernel Variants and RAJA Features]-----
  a. C++ style nested for loops
  b. RAJA style outer loop with a sequential policy
     and a C++ style inner for loop
  c. RAJA style nested for loops with sequential policies
  d. RAJA forallN loop with sequential policies
     i. This kernel introduces RAJA::ExecList
  e. RAJA forallN loop with OpenMP parallism on the outer loop
  f. RAJA forallN loop executed on the CUDA API
     i.  This kernel illustrates constructing two-dimensional thread blocks
         for use of the CUDA execution policy.
     ii. The current implementation of forallN using the CUDA
         variant is performed asynchronously and thus a barrier
         (cudaDeviceSynchronize) is placed after calling forallN.
*/

/*
  ---[Constant values]----
  N   - Defines the number of rows/columns in a matrix
  NN  - Total number of entries in a matrix
  DIM - Dimension of the data structure in which the matrices
        are stored

  CUDA_BLOCK_SIZE_X - Number of threads in the
                      x-dimension of a cuda thread block

  CUDA_BLOCK_SIZE_Y - Number of threads in the
                      y-dimension of a cuda thread block
*/
const int N = 1000;
const int NN = N * N;
const int DIM = 2;

const int CUDA_BLOCK_SIZE_X = 16;
const int CUDA_BLOCK_SIZE_Y = 16;

/*
 Macros are used here to simplify indexing
*/
#define A(x1, x2) A[x1 + N * x2]
#define B(x1, x2) B[x1 + N * x2]
#define C(x1, x2) C[x1 + N * x2]

template <typename T>
void checkSolution(T *C, int N);

template <typename T>
void checkSolution(RAJA::View<T, RAJA::Layout<DIM>> Cview, int N);

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  printf("Example 2: Multiplying Two N x N Matrices \n \n");
  double *A = memoryManager::allocate<double>(NN);
  double *B = memoryManager::allocate<double>(NN);
  double *C = memoryManager::allocate<double>(NN);

  /*
    Intialize matrices
   */
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      A(row, col) = row;
      B(row, col) = col;
    }
  }

  printf("Standard C++ Nested Loops \n");
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {

      double dot = 0.0;
      for (int k = 0; k < N; ++k) {
        dot += A(row, k) * B(k, col);
      }

      C(row, col) = dot;
    }
  }
  checkSolution<double>(C, N);

  /*
    As an alternative to marcos RAJA::View wraps
    a pointer to enable multi-dimensional indexing
    In this example our data is assumed to be two-dimensional
    with N values in each component.
  */
  RAJA::View<double, RAJA::Layout<DIM>> Aview(A, N, N);
  RAJA::View<double, RAJA::Layout<DIM>> Bview(B, N, N);
  RAJA::View<double, RAJA::Layout<DIM>> Cview(C, N, N);

  /*
    As the loops use the same bounds, we may specify
    the bounds prior to the use of any RAJA loops
  */
  RAJA::RangeSegment matBounds(0, N);


  printf("RAJA: Forall - Sequential Policies\n");
  RAJA::forall<RAJA::seq_exec>(
    matBounds, [=](RAJA::Index_type row) {    

      for (int col = 0; col < N; ++col) {
        
        double dot = 0.0;
        for (int k = 0; k < N; ++k) {
          dot += Aview(row, k) * Bview(k, col);
        }
        
        Cview(row, col) = dot;
      }
      
    });
  checkSolution<double>(Cview, N);

  printf("RAJA: Nested Forall - Sequential Policies\n");
  /*
    Forall loops may be nested under sequential and omp policies
  */
  RAJA::forall<RAJA::seq_exec>(
    matBounds, [=](RAJA::Index_type row) {

      RAJA::forall<RAJA::seq_exec>(
        matBounds, [=](RAJA::Index_type col) {
        

          double dot = 0.0;
          for (int k = 0; k < N; ++k) {
            dot += Aview(row, k) * Bview(k, col);
          }

          Cview(row, col) = dot;
        });
    });
  checkSolution<double>(Cview, N);


  printf("RAJA: ForallN - Sequential Policies\n");
  /*
    Nested forall loops may be collapsed into a single forallN loop
  */
  RAJA::forallN<RAJA::NestedPolicy<
    RAJA::ExecList<RAJA::seq_exec,RAJA::seq_exec>>>(    
      matBounds, matBounds, [=](RAJA::Index_type row, RAJA::Index_type col) {
      
        double dot = 0.0;
        for (int k = 0; k < N; ++k) {
          dot += Aview(row, k) * Bview(k, col);
        }
        
        Cview(row, col) = dot;
      });
  checkSolution<double>(Cview, N);
  

#if defined(RAJA_ENABLE_OPENMP)
  printf("RAJA: ForallN - OpenMP/Sequential Policies\n");
  /*
    Here the outer loop is excuted in parallel while the inner loop
    is executed sequentially
  */
  RAJA::forallN<RAJA::NestedPolicy<
    RAJA::ExecList<RAJA::omp_parallel_for_exec,RAJA::seq_exec>>>(
      matBounds, matBounds, [=](RAJA::Index_type row, RAJA::Index_type col) {
      
        double dot = 0.0;
        for (int k = 0; k < N; ++k) {
          dot += Aview(row, k) * Bview(k, col);
        }

        Cview(row, col) = dot;
      });
  checkSolution<double>(Cview, N);
#endif


#if defined(RAJA_ENABLE_CUDA)
  printf("RAJA: ForallN - CUDA Policies\n");
  /*
    This example illustrates creating two-dimensional thread blocks as described
    under the CUDA nomenclature
  */
  RAJA::forallN<RAJA::NestedPolicy<RAJA::
    ExecList<RAJA::cuda_threadblock_y_exec<CUDA_BLOCK_SIZE_X>,    
      RAJA::cuda_threadblock_x_exec<CUDA_BLOCK_SIZE_Y>>>>(   
        matBounds, matBounds, [=] __device__(RAJA::Index_type row, RAJA::Index_type col) {
        
          double dot = 0.0;
          for (int k = 0; k < N; ++k) {
            dot += Aview(row, k) * Bview(k, col);
          }
          
          Cview(row, col) = dot;
        });
  cudaDeviceSynchronize();
  checkSolution<double>(Cview, N);
#endif

  memoryManager::deallocate(A);
  memoryManager::deallocate(B);
  memoryManager::deallocate(C);

  return 0;
}

/*
  Function which checks for correctness
*/
template <typename T>
void checkSolution(RAJA::View<T, RAJA::Layout<DIM>> Cview, int in_N)
{

  RAJA::forall<RAJA::seq_exec>(
    RAJA::RangeSegment(0, N), [=](RAJA::Index_type row) {
    
      RAJA::forall<RAJA::seq_exec>(
        RAJA::RangeSegment(0, N), [=](RAJA::Index_type col) {
        
          double diff = Cview(row, col) - row * col * in_N;
            
          if (std::abs(diff) > 1e-9) {
            printf("Incorrect Result \n \n");
            return;
          }

        });
    });
  printf("Correct Result \n \n");
};

template <typename T>
void checkSolution(T *C, int in_N)
{

  RAJA::forall<RAJA::seq_exec>(
    RAJA::RangeSegment(0, N), [=](RAJA::Index_type row) {    
                                 
      RAJA::forall<RAJA::seq_exec>(
        RAJA::RangeSegment(0, N), [=](RAJA::Index_type col) {       

          double diff = C(row, col) - row * col * in_N;
            
          if (std::abs(diff) > 1e-9) {
            printf("Incorrect Result \n \n");
            return;
          }

        });
    });
  printf("Correct Result \n \n");
};
