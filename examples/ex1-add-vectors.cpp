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
#include <cstring>
#include <iostream>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

/*
 *  Vector Addition Example
 *
 *  Computes C = A + B, where A, B, C are vectors of ints.
 *  It illustrates similarities between a  C-style for-loop and a RAJA 
 *  forall loop.
 *
 *  RAJA features shown:
 *    - `forall` loop iteration template method
 *    -  Index range segment
 *    -  Execution policies
 */

/*
  CUDA_BLOCK_SIZE - specifies the number of threads in a CUDA thread block
*/
#if defined(RAJA_ENABLE_CUDA)
const int CUDA_BLOCK_SIZE = 256;
#endif

//
//  Function to compare solution to reference and print result P/F.
//
void checkSolution(int *C, int len) 
{
  bool correct = true;
  for (int i = 0; i < len; i++) {
    if ( C[i] != 0 ) { correct = false; }
  }
  if ( correct ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
}

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA vector addition example...\n";

  const int N = 1000;
  int *A = memoryManager::allocate<int>(N);
  int *B = memoryManager::allocate<int>(N);
  int *C = memoryManager::allocate<int>(N);

  for (int i = 0; i < N; ++i) {
    A[i] = -i;
    B[i] = i;
  }


  std::cout << "\n Running C-version of vector addition...\n";

  for (int i = 0; i < N; ++i) {
    C[i] = A[i] + B[i];
  }

  checkSolution(C, N);


  std::cout << "\n Running RAJA sequential vector addition...\n";

  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, N), [=] (int i) { 
    C[i] = A[i] + B[i]; 
  });    

  checkSolution(C, N);


  std::cout << "\n Running RAJA SIMD vector addition...\n";

  RAJA::forall<RAJA::simd_exec>(RAJA::RangeSegment(0, N), [=] (int i) { 
    C[i] = A[i] + B[i]; 
  });    

  checkSolution(C, N);



#if defined(RAJA_ENABLE_OPENMP)
  std::cout << "\n Running RAJA OpenMP vector addition...\n";

  RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(0, N), [=] (int i) { 
    C[i] = A[i] + B[i]; 
  });    

  checkSolution(C, N);
#endif


#if defined(RAJA_ENABLE_CUDA)
  std::cout << "\n Running RAJA CUDA vector addition...\n";

  RAJA::forall<RAJA::cuda_exec<CUDA_BLOCK_SIZE>>(RAJA::RangeSegment(0, N), 
    [=] RAJA_DEVICE (int i) { 
    C[i] = A[i] + B[i]; 
  });    

  checkSolution(C, N);
#endif

  memoryManager::deallocate(A);
  memoryManager::deallocate(B);
  memoryManager::deallocate(C);

  std::cout << "\n DONE!...\n";

  return 0;
}
