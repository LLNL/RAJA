//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <iostream>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"
#include "RAJA/util/defines.hpp"

/*
  Example 1: Adding Two Vectors

  ----[Details]---------------------
  Starting with a C++ style for loop, this example illustrates
  how to construct RAJA versions of the same loop with different
  execution policies.

  In this example, three integer arrays (A,B,C) are allocated
  using the templated memory manager found in this folder.
  The vectors A and B are initialized to have opposite values
  and thus when the entries are added the result should be zero.
  The result of the vector addition is stored in C. The function
  checkSolution is used to verify correctness.

  -----[RAJA Concepts]---------------
  1. Introduction of the forall loop and basic RAJA policies

  RAJA::forall<exec_policy>(iter_space I, [=] (index_type i)) {

         //body

  });

  [=] By-copy capture
  [&] By-reference capture (for non-unified memory targets)
  exec_policy - Specifies how the traversal occurs
  iter_space  - Iteration space for RAJA loop (any random access container is
  expected)
  index_type  - Index for RAJA loops

  ----[Kernel Variants and RAJA Features]------------
  a. C++ style for loop
  b. RAJA style for loop with sequential iterations
     i.  Introduces the seq_exec policy
     ii. Introduces RAJA::RangeSegment
  c. RAJA style for loop with omp parallelism
     i. Introduces the omp_parallel_for_exec policy
  d. RAJA style for loop with CUDA parallelism
     i. Introduces the cuda_exec policy
 */

/*
  CUDA_BLOCK_SIZE - specifies the number of threads in a CUDA thread block
*/
const int CUDA_BLOCK_SIZE = 256;

/*
  Function to verify correctness
*/
void checkSolution(int *C, int in_N);

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  printf("Example 1: Adding Two Vectors \n \n");

  const int N = 1000;
  int *A = memoryManager::allocate<int>(N);
  int *B = memoryManager::allocate<int>(N);
  int *C = memoryManager::allocate<int>(N);

  for (int i = 0; i < N; ++i) {
    A[i] = -i;
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

    RAJA::RangeSegment(start,stop) - Generates a contiguous sequence of numbers
    by the [start, stop) interval specified
  */
  RAJA::forall<RAJA::seq_exec>(
    RAJA::RangeSegment(0, N), [=](RAJA::Index_type i) { 

      C[i] = A[i] + B[i]; 

    });    
  checkSolution(C, N);


#if defined(RAJA_ENABLE_OPENMP)
  printf("RAJA: OpenMP Policy \n");
  /*
    RAJA::omp_parallel_for_exec - executes the forall loop using the
    #pragma omp parallel for directive
  */
  RAJA::forall<RAJA::omp_parallel_for_exec>(
    RAJA::RangeSegment(0, N), [=](RAJA::Index_type i) {
    
      C[i] = A[i] + B[i];

    });
  checkSolution(C, N);
#endif


#if defined(RAJA_ENABLE_CUDA)
  printf("RAJA: CUDA Policy \n");
  /*
    RAJA::cuda_exec<CUDA_BLOCK_SIZE> - excecutes loop using the CUDA API
    Here the __device__ keyword is used to specify a CUDA kernel
  */
  RAJA::forall<RAJA::cuda_exec<CUDA_BLOCK_SIZE>>
    (RAJA::RangeSegment(0, N), [=] __device__(RAJA::Index_type i) { 
      
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
  Function to check for correctness
*/
void checkSolution(int *C, int in_N)
{

  RAJA::forall<RAJA::seq_exec>
    (RAJA::RangeSegment(0, in_N), [=](RAJA::Index_type i) {
     
      if (std::abs(C[i]) != 0) {
        printf("Error in Result \n \n");
        return;
      }

  });
  
  printf("Correct Result \n \n");
}
