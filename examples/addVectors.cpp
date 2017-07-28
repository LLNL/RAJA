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

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"
#include "RAJA/util/defines.hpp"

const int CUDA_BLOCK_SIZE = 256;

void checkSolution(int *C, int in_N);

/*
  Example 1: Adding Two Vectors

  ----[Details]---------------------
  Adds two vectors of length N

  -----[RAJA Concepts]---------------
  1. Introduces the forall loop and basic RAJA policies

  RAJA::forall<RAJA::exec_policy>(RAJA::Range,[=](index i)) {

         //body

  });

  [=] Pass by copy
  [&] Pass by reference
  RAJA::exec_policy - specifies how the traversal occurs
  RAJA::Range       - Provides a list of iterables for an index of the loop
  RAJA::Index_type  - Index for RAJA loops
 */
int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{
  
  printf("Example 1: Adding Two Vectors \n \n");
  
  const int N = 1000;
  int *A = memoryManager::allocate<int>(N);
  int *B = memoryManager::allocate<int>(N);
  int *C = memoryManager::allocate<int>(N);

  for (int i = 0; i < N; ++i) {
    A[i] = i;
    B[i] = i;
  }

  printf("Standard C++ Loop \n");
  for (int i = 0; i < N; ++i) {
    C[i] = A[i] + B[i];
  }
  checkSolution(C, N);


  printf("RAJA: Sequential Policy \n");
  /*
    RAJA::seq_exec -  Executes the loop sequentially

    RAJA::RangeSegment(start,stop) - generates a contiguous sequence of numbers
    by the [start, stop) interval specified 
  */
  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, N),
                               [=](RAJA::Index_type i) { C[i] = A[i] + B[i]; });
  checkSolution(C, N);


#if defined(RAJA_ENABLE_OPENMP)
  printf("RAJA: OpenMP Policy \n");
  /*
    RAJA::omp_parallel_for_exec - executes the forall loop using the
    #pragma omp parallel for directive
  */
  RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(0, N),
                                            [=](RAJA::Index_type i) { C[i] = A[i] + B[i]; });
  checkSolution(C, N);
#endif


#if defined(RAJA_ENABLE_CUDA)

  printf("RAJA: CUDA Policy \n");
  /*
    RAJA::cuda_exec<CUDA_BLOCK_SIZE> - excecutes the forall loop using the CUDA
    API. Each thread is assigned to an iteration of the loop

    CUDA_BLOCK_SIZE - specifies the number of threads in a CUDA thread block
  */
  RAJA::forall<RAJA::cuda_exec<CUDA_BLOCK_SIZE>>(RAJA::RangeSegment(0, N),
                                                 [=] __device__(RAJA::Index_type i) {
                                                   C[i] = A[i] + B[i];
                                                 });
  checkSolution(C, N);
#endif

  memoryManager::deallocate(A);
  memoryManager::deallocate(B);
  memoryManager::deallocate(C);

  return 0;
}


/*
  RAJA can also be used outside of computational kernels
*/
void checkSolution(int *C, int in_N)
{


  RAJA::forall<RAJA::seq_exec>(
  RAJA::RangeSegment(0, in_N), [=](RAJA::Index_type i) {
                                 
    if ( abs(C[i]-(i+i) ) != 0) {
      printf("Error in Result \n \n");
      return;      
    }

  });
 
  printf("Correct Result \n \n");
}
