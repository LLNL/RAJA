//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "RAJA/RAJA.hpp"

#include "memoryManager.hpp"

/*
 *  EXERCISE #1: Vector Addition
 *
 *  In this exercise, you will compute c = a + b, where a, b, c are 
 *  integer vectors.
 *
 *  This file contains sequential and OpenMP variants of the vector addition
 *  using C-style for-loops. You will fill in RAJA versions of these variants,
 *  plus a RAJA CUDA version if you have access to an NVIDIA GPU and a CUDA
 *  compiler, in empty code sections indicated by comments.
 *
 *  The exercise shows you how to use RAJA in its simplest form and 
 *  illustrates similarities between a C-style for-loop and a RAJA forall loop.
 *
 *  RAJA features you will use:
 *    - `forall` loop iteration template method
 *    -  Index range segment
 *    -  Execution policies
 *
 * Note: if CUDA is enabled, CUDA unified memory is used. 
 */

/*
  CUDA_BLOCK_SIZE - specifies the number of threads in a CUDA thread block
*/
#if defined(RAJA_ENABLE_CUDA)
const int CUDA_BLOCK_SIZE = 256;
#endif

//
// Functions for checking and printing arrays
//
void checkResult(int* c, int* c_ref, int len); 
void printArray(int* v, int len);


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nExercise #1: RAJA Vector Addition...\n";

//
// Define vector length
//
  const int N = 1000000;

//
// Allocate and initialize vector data to random numbers in [1, 10].
//
  int *a = memoryManager::allocate<int>(N);
  int *b = memoryManager::allocate<int>(N);
  int *c = memoryManager::allocate<int>(N);
  int *c_ref = memoryManager::allocate<int>(N);

  for (int i = 0; i < N; ++i) {
    a[i] = rand() % 10 + 1;
    b[i] = rand() % 10 + 1;
  }


//----------------------------------------------------------------------------//
// C-style sequential variant establishes reference solution to compare with.
//----------------------------------------------------------------------------//

  std::memset(c_ref, 0, N * sizeof(int));

  std::cout << "\n Running C-style sequential vector addition...\n";

  for (int i = 0; i < N; ++i) {
    c_ref[i] = a[i] + b[i];
  }

//printArray(c_ref, N);


//----------------------------------------------------------------------------//
// RAJA::seq_exec policy enforces strictly sequential execution.
//----------------------------------------------------------------------------//

  std::memset(c, 0, N * sizeof(int));

  std::cout << "\n Running RAJA sequential vector addition...\n";

  using EXEC_POL1 = RAJA::seq_exec;

  RAJA::forall< EXEC_POL1 >(RAJA::RangeSegment(0, N), [=] (int i) { 
    c[i] = a[i] + b[i]; 
  });    

  checkResult(c, c_ref, N);
//printArray(c, N);


//----------------------------------------------------------------------------//
// RAJA::simd_exec policy attempts to force the compiler to generate SIMD
// vectorization optimizations.
//----------------------------------------------------------------------------//

  std::memset(c, 0, N * sizeof(int));

  std::cout << "\n Running RAJA SIMD vector addition...\n";

  using EXEC_POL2 = RAJA::simd_exec;

  RAJA::forall< EXEC_POL2 >(RAJA::RangeSegment(0, N), [=] (int i) { 
    c[i] = a[i] + b[i]; 
  });    

  checkResult(c, c_ref, N);
//printArray(c, N);


//----------------------------------------------------------------------------//
// RAJA::loop_exec policy allows the compiler to generate optimizations 
// (e.g., SIMD) if it thinks it is safe to do so.
//----------------------------------------------------------------------------//

  std::memset(c, 0, N * sizeof(int));

  std::cout << "\n Running RAJA loop-exec vector addition...\n";

  using EXEC_POL3 = RAJA::loop_exec;

  RAJA::forall< EXEC_POL3 >(RAJA::RangeSegment(0, N), [=] (int i) { 
    c[i] = a[i] + b[i];
  });

  checkResult(c, c_ref, N);
//printArray(c, N);


//----------------------------------------------------------------------------//
// C-style OpenMP multithreading variant.
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)

  std::memset(c, 0, N * sizeof(int));

  std::cout << "\n Running C-style OpenMP vector addition...\n";

  #pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    c[i] = a[i] + b[i];
  }

  checkResult(c, c_ref, N);
//printArray(c, N);

#endif


//----------------------------------------------------------------------------//
// RAJA::omp_parallel_for_exec policy runs the loop in parallel using
// OpenMP multithreading.
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)

  std::memset(c, 0, N * sizeof(int));

  std::cout << "\n Running RAJA OpenMP multithreaded vector addition...\n";

  using EXEC_POL4 = RAJA::omp_parallel_for_exec;

  RAJA::forall< EXEC_POL4 >(RAJA::RangeSegment(0, N), [=] (int i) { 
    c[i] = a[i] + b[i]; 
  });    

  checkResult(c, c_ref, N);
//printArray(c, N);
#endif


//----------------------------------------------------------------------------//
// RAJA::cuda_exec policy runs the loop as a CUDA kernel on a GPU device.
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)

  std::memset(c, 0, N * sizeof(int));

  std::cout << "\n Running RAJA CUDA vector addition...\n";

  using EXEC_POL5 = RAJA::cuda_exec<CUDA_BLOCK_SIZE>;

  RAJA::forall< EXEC_POL5 >(RAJA::RangeSegment(0, N), [=] RAJA_DEVICE (int i) {
    c[i] = a[i] + b[i];
  });

  checkResult(c, c_ref, N);
//printArray(c, N);
#endif

//----------------------------------------------------------------------------//

//
// Clean up.
//
  memoryManager::deallocate(a);
  memoryManager::deallocate(b);
  memoryManager::deallocate(c);
  memoryManager::deallocate(c_ref);

  std::cout << "\n DONE!...\n";

  return 0;
}

//
// Function to check result and report P/F.
//
void checkResult(int* c, int* c_ref, int len)
{
  bool correct = true;
  for (int i = 0; i < len; i++) {
    if ( correct && c[i] != c_ref[i] ) { correct = false; }
  }
  if ( correct ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
}

//
// Function to print array.
//
void printArray(int* v, int len)
{
  std::cout << std::endl;
  for (int i = 0; i < len; i++) {
    std::cout << "v[" << i << "] = " << v[i] << std::endl;
  }
  std::cout << std::endl;
}

