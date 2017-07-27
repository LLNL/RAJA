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
#include <algorithm>
#include <initializer_list>

#include "RAJA/RAJA.hpp"
#include "RAJA/index/RangeSegment.hpp"
#include "RAJA/util/defines.hpp"

#include "memoryManager.hpp"

const int DIM = 2;
void checkSolution(double *C, int in_N);
void checkSolution(RAJA::View<double,RAJA::Layout<DIM> > Cview, int in_N);

/*
  Example 2: Multiplying Two Matrices

  ----[Details]--------------------
  Multiplies two N x N matrices

  -----[RAJA Concepts]-------------
  1. Nesting of forall loops (Not currently supported in CUDA)
  2. ForallN variant of the RAJA loop
  3. RAJA View
*/
int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  printf("Example 2: Multiplying Two N x N Matrices \n \n");
  const int N = 1000;
  const int NN = N * N;

  double *A = memoryManager::allocate<double>(NN);
  double *B = memoryManager::allocate<double>(NN);
  double *C = memoryManager::allocate<double>(NN);

  for(int i=0; i<NN; ++i){
    A[i] = 1.0; 
    B[i] = 1.0; 
  }

  
  printf("Standard C++ Loop \n");
  for (int r = 0; r < N; ++r) {
    for (int c = 0; c < N; ++c) {

      int cId = c + r * N;
      double dot = 0.0;

      for (int k = 0; k < N; ++k) {

        int aId = k + r * N;
        int bId = c + k * N;
        dot += A[aId] * B[bId];
      }

      C[cId] = dot;
    }
  }
  checkSolution(C, N);

  /*
    RAJA::View - Introduces Multidimensional arrays
  */  

  RAJA::View<double, RAJA::Layout<DIM> > Aview(A, N, N);
  RAJA::View<double, RAJA::Layout<DIM> > Bview(B, N, N);
  RAJA::View<double, RAJA::Layout<DIM> > Cview(C, N, N);


  printf("RAJA: Sequential Policy - Single forall \n");  
  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, N), [=](int row) {
      for(int col = 0; col < N; ++col){
        
        double dot = 0.0;
        for (int k = 0; k < N; ++k) {
          dot += Aview(row,k)*Bview(k,col);
        }
        
        Cview(row,col) = dot; 
      }

    });
  checkSolution(Cview, N);


  printf("RAJA: Sequential Policy - Nested forall \n");
  /*
    Forall loops may be nested under sequential and omp policies
  */
  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, N), [=](int row) {
    RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, N), [=](int col) {

      double dot = 0.0;
      for (int k = 0; k < N; ++k) {
        dot += Aview(row,k)*Bview(k,col);
      }
      
      Cview(row,col) = dot;
    });
  });
  checkSolution(Cview, N);

  printf("RAJA: Sequential Policy RAJA - forallN \n");
  /*
    Nested forall loops may be collapsed into a single forallN loop
  */
  RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec,
                                                  RAJA::seq_exec>>>(
  RAJA::RangeSegment(0,N), RAJA::RangeSegment(0,N), [=](int row, int col) {

        double dot = 0.0;
        for (int k = 0; k < N; ++k) {

          dot += Aview(row,k) * Bview(k,col);
        }

        Cview(row,col) = dot;
      });
  checkSolution(Cview, N);
  

#if defined(RAJA_ENABLE_OPENMP)
  printf("RAJA: OpenMP/Sequential Policy - forallN \n");
  /*
    Here the outer loop is excuted in parallel while the inner loop
    is executed sequentially
  */
  RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec,
                                                  RAJA::seq_exec>>>(
      RAJA::RangeSegment(0, N), RAJA::RangeSegment(0, N), [=](int row, int col) {

        double dot = 0.0;

        for (int k = 0; k < N; ++k) {
          
          dot += Aview(row,col) * Bview(row,col);
        }
        Cview(row,col) = dot;
      });
  checkSolution(C, N);
#endif


#if defined(RAJA_ENABLE_CUDA)
  printf("RAJA: CUDA Policy - forallN \n");
  /*
    This example illustrates creating two-dimensional thread blocks as described
    under the CUDA nomenclature
  */
  RAJA::forallN<RAJA::NestedPolicy<
    RAJA::ExecList<RAJA::cuda_threadblock_y_exec<16>,    
    RAJA::cuda_threadblock_x_exec<16>>>>(
          RAJA::RangeSegment(0, N),
          RAJA::RangeSegment(0, N),
          [=] __device__(int row, int col) {
            
            double dot = 0.0;

            for (int k = 0; k < N; ++k) {

              dot += Aview(row,k) * Bview(k,col);
            }
            Cview(row,col) = dot; 
          });
  cudaDeviceSynchronize();
  checkSolution(Cview, N);
#endif


  memoryManager::deallocate(A);
  memoryManager::deallocate(B);
  memoryManager::deallocate(C);

  return 0;
}

void checkSolution(double *C, int in_N)
{

  for (int id = 0; id < in_N * in_N; ++id) {
    if (abs(C[id] - in_N) > 1e-9) {
      printf("Error in Result \n \n");
      return;
    }
  }
  printf("Correct Result \n \n");
}

void checkSolution(RAJA::View<double,RAJA::Layout<DIM> > Cview, int in_N){

  for(int row = 0; row < in_N; ++row) {
    for(int col = 0; col < in_N; ++col) {

      if (abs(Cview(row,col) - in_N) > 1e-9) {
        printf("Error in Result \n \n");
        return;
      }

    }
  }

  printf("Correct Result \n \n");
};
