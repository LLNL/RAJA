//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
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
#include <cstring>
#include <iostream>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

/*
 *  Vector Addition Example
 *
 *  Computes c = a + b, where a, b, c are vectors of ints.
 *  It illustrates similarities between a  C-style for-loop and a RAJA 
 *  forall loop.
 *
 *  RAJA features shown:
 *    - `forall` loop iteration template method
 *    -  Index range segment
 *    -  Execution policies
 *
 * If CUDA is enabled, CUDA unified memory is used. 
 */

/*
  CUDA_BLOCK_SIZE - specifies the number of threads in a CUDA thread block
*/
#if defined(RAJA_ENABLE_CUDA)
const int CUDA_BLOCK_SIZE = 256;
#endif

//
// Functions for checking and printing results
//
void checkResult(int* res, int len); 
void printResult(int* res, int len);


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA vector addition example...\n";

//
// Define vector length
//
  const int N = 1000000;

//
// Allocate and initialize vector data
//
  int *a = memoryManager::allocate<int>(N);
  int *b = memoryManager::allocate<int>(N);
  int *c = memoryManager::allocate<int>(N);

  for (int i = 0; i < N; ++i) {
    a[i] = -i;
    b[i] = i;
  }


//----------------------------------------------------------------------------//

  std::cout << "\n Running C-version of vector addition...\n";

  for (int i = 0; i < N; ++i) {
    c[i] = a[i] + b[i];
  }

  checkResult(c, N);
//printResult(c, N);


//----------------------------------------------------------------------------//
// RAJA::seq_exec policy enforces strictly sequential execution.... 
//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA sequential vector addition...\n";

  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, N), [=] (int i) { 
    c[i] = a[i] + b[i]; 
  });    

  checkResult(c, N);
//printResult(c, N);


//----------------------------------------------------------------------------//
// RAJA::simd_exec policy should force the compiler to generate SIMD
// vectorization optimizations.... 
//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA SIMD vector addition...\n";

  RAJA::forall<RAJA::simd_exec>(RAJA::RangeSegment(0, N), [=] (int i) { 
    c[i] = a[i] + b[i]; 
  });    

  checkResult(c, N);
//printResult(c, N);


//----------------------------------------------------------------------------//
// RAJA::loop_exec policy means that the compiler is allowed to generate 
// optimizations (e.g., SIMD) if it thinks it is safe to do so...
//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA loop-exec vector addition...\n";

  RAJA::forall<RAJA::loop_exec>(RAJA::RangeSegment(0, N), [=] (int i) {
    c[i] = a[i] + b[i];
  });

  checkResult(c, N);
//printResult(c, N);


//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)
  std::cout << "\n Running RAJA OpenMP vector addition...\n";

  RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(0, N), [=] (int i) { 
    c[i] = a[i] + b[i]; 
  });    

  checkResult(c, N);
//printResult(c, N);
#endif


//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)
  std::cout << "\n Running RAJA CUDA vector addition...\n";

  RAJA::forall<RAJA::cuda_exec<CUDA_BLOCK_SIZE>>(RAJA::RangeSegment(0, N), 
    [=] RAJA_DEVICE (int i) { 
    c[i] = a[i] + b[i]; 
  });    

  checkResult(c, N);
//printResult(c, N);
#endif

//----------------------------------------------------------------------------//

//
// Clean up.
//
  memoryManager::deallocate(a);
  memoryManager::deallocate(b);
  memoryManager::deallocate(c);

  std::cout << "\n DONE!...\n";

  return 0;
}

//
// Function to check result and report P/F.
//
void checkResult(int* res, int len) 
{
  bool correct = true;
  for (int i = 0; i < len; i++) {
    if ( res[i] != 0 ) { correct = false; }
  }
  if ( correct ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
}

//
// Function to print result.
//
void printResult(int* res, int len)
{
  std::cout << std::endl;
  for (int i = 0; i < len; i++) {
    std::cout << "result[" << i << "] = " << res[i] << std::endl;
  }
  std::cout << std::endl;
}

