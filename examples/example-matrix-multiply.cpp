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

// Matrix dimensions are assumed to be N x N
const int N = 1000;
const int NN = N * N;

const int DIM = 2;

template <typename T>
void checkSolution(RAJA::View<T, RAJA::Layout<DIM>> Cview, int in_N);

/*
  Example 2: Multiplying Two Matrices

  ----[Details]--------------------
  This example illustrates how RAJA may be
  gradually introduced into existing codes


  -----[RAJA Concepts]-------------
  1. Nesting forall loops (Not currently supported in CUDA)

  2. ForallN loop

  RAJA::forallN<
  RAJA::NestedPolicy<RAJA::exec_policy, .... , RAJA::exec_policy> >
  (RAJA::Range1,..., RAJA::RangeN, [=](RAJA::Index_type i1,..., RAJA::Index_type iN) {

         //body

  });

  [=] Pass by copy
  [&] Pass by reference
  RAJA::NestedPolicy - List of execution policies for the loops
  RAJA::exec_policy  - Specifies how the traversal occurs
  RAJA::Range - List of iterables for an index of the loop

  3. RAJA::View - RAJA's wrapper for multidimensional indexing

*/
int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  printf("Example 2: Multiplying Two N x N Matrices \n \n");
  double *A = memoryManager::allocate<double>(NN);
  double *B = memoryManager::allocate<double>(NN);
  double *C = memoryManager::allocate<double>(NN);

  for (int i = 0; i < NN; ++i) {
    A[i] = 1.0;
    B[i] = 1.0;
  }

  /*
    RAJA::View - RAJA's wrapper for multidimensional indexing
  */

  RAJA::View<double, RAJA::Layout<DIM>> Aview(A, N, N);
  RAJA::View<double, RAJA::Layout<DIM>> Bview(B, N, N);
  RAJA::View<double, RAJA::Layout<DIM>> Cview(C, N, N);

  printf("Standard C++ Loop \n");
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {

      double dot = 0.0;
      for (int k = 0; k < N; ++k) {
        dot += Aview(row, k) * Bview(k, col);
      }

      Cview(row, col) = dot;
    }
  }
  checkSolution<double>(Cview, N);

  /*
    RAJA bounds may be specified prior to forall statements
  */
  RAJA::RangeSegment matBounds(0, N);

  printf("RAJA: Sequential Policy - Single forall \n");
  RAJA::forall<RAJA::seq_exec>
    (matBounds, [=](RAJA::Index_type row) {
  
    for (int col = 0; col < N; ++col) {

      double dot = 0.0;
      for (int k = 0; k < N; ++k) {
        dot += Aview(row, k) * Bview(k, col);
      }
      
      Cview(row, col) = dot;
    }
    
  });
  checkSolution<double>(Cview, N);

  printf("RAJA: Sequential Policy - Nested forall \n");
  /*
    Forall loops may be nested under sequential and omp policies
  */
  RAJA::forall<RAJA::seq_exec>
    (matBounds, [=](RAJA::Index_type row) {  
      
      RAJA::forall<RAJA::seq_exec>
        (matBounds, [=](RAJA::Index_type col) {
          
          double dot = 0.0;
          for (int k = 0; k < N; ++k) {
            dot += Aview(row, k) * Bview(k, col);
          }
          
          Cview(row, col) = dot;
        });
    });
  checkSolution<double>(Cview, N);
  
  
  printf("RAJA: Sequential Policy RAJA - forallN \n");
  /*
    Nested forall loops may be collapsed into a single forallN loop
  */
  RAJA::forallN<RAJA::NestedPolicy
    <RAJA::ExecList<RAJA::seq_exec, RAJA::seq_exec>>>
    (matBounds, matBounds, [=](RAJA::Index_type row, RAJA::Index_type col) {
      
      double dot = 0.0;
      for (int k = 0; k < N; ++k) {        
        dot += Aview(row, k) * Bview(k, col);
      }
      
      Cview(row, col) = dot;
    });
  checkSolution<double>(Cview, N);
  
  
#if defined(RAJA_ENABLE_OPENMP)
  printf("RAJA: OpenMP/Sequential Policy - forallN \n");
  /*
    Here the outer loop is excuted in parallel while the inner loop
    is executed sequentially
  */
  RAJA::forallN<RAJA::NestedPolicy
    <RAJA::ExecList<RAJA::omp_parallel_for_exec, RAJA::seq_exec>>>
    (matBounds, matBounds, [=](RAJA::Index_type row, RAJA::Index_type col) {
      
      double dot = 0.0;
      for (int k = 0; k < N; ++k) {
        dot += Aview(row, k) * Bview(k, col);
      }
      
      Cview(row, col) = dot;
    });
  checkSolution<double>(Cview, N);
#endif
  
  
#if defined(RAJA_ENABLE_CUDA)
  printf("RAJA: CUDA Policy - forallN \n");
  /*
    This example illustrates creating two-dimensional thread blocks as described
    under the CUDA nomenclature
  */
  RAJA::forallN<RAJA::NestedPolicy
    <RAJA::ExecList<RAJA::cuda_threadblock_y_exec<16>, RAJA::cuda_threadblock_x_exec<16>>>>
    (matBounds, matBounds, [=] __device__(RAJA::Index_type row, RAJA::Index_type col) { 
      
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

template <typename T>
void checkSolution(RAJA::View<T, RAJA::Layout<DIM>> Cview, int in_N)
{

  for (int row = 0; row < in_N; ++row) {
    for (int col = 0; col < in_N; ++col) {
      
      double diff = Cview(row,col) - in_N;
      
      if (abs(diff) > 1e-9) {
        return;
      }
    }
  }
  printf("Correct Result \n \n");
};
