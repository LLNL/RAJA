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
 *  Vector Dot Product Example
 *
 *  Computes dot = (A,B), where A, B are vectors of 
 *  doubles and dot is a scalar double. It illustrates how RAJA
 *  supports a portable parallel reduction opertion in a way that 
 *  the code looks like it does in a sequential implementation.
 *
 *  RAJA features shown:
 *    - `forall` loop iteration template method
 *    -  Index range segment
 *    -  Execution policies
 *    -  Reduction types
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
//  Function to compare computed dot product to expected value
//
void checkSolution(double compdot, double refdot)
{
  if ( compdot == refdot ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
}

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA vector dot product example...\n";

  const int N = 1000;
  int *A = memoryManager::allocate<int>(N);
  int *B = memoryManager::allocate<int>(N);

  for (int i = 0; i < N; ++i) {
    A[i] = 1.0;
    B[i] = 1.0;
  }

  double dot = 0.0;

//
// C-style dot product operation.
//
  std::cout << "\n Running C-version of dot product...\n";

  for (int i = 0; i < N; ++i) {
    dot += A[i] * B[i];
  }

  checkSolution(dot, N);


//
// RAJA version of sequential dot product.
//
  std::cout << "\n Running RAJA sequential dot product...\n";

  RAJA::ReduceSum<RAJA::seq_reduce, double> seqdot(0.0);

  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, N), [=] (int i) { 
    seqdot+= A[i] * B[i]; 
  });

  dot = seqdot.get();

  checkSolution(dot, N);


#if defined(RAJA_ENABLE_OPENMP)
//
// RAJA version of SIMD dot product.
//
  std::cout << "\n Running RAJA OpenMP dot product...\n";

  RAJA::ReduceSum<RAJA::omp_reduce, double> ompdot(0.0);

  RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(0, N), [=] (int i) { 
    ompdot += A[i] * B[i]; 
  });    

  dot = ompdot.get();

  checkSolution(dot, N);
#endif


#if defined(RAJA_ENABLE_CUDA)
//
// RAJA version of CUDA dot product.
//
  std::cout << "\n Running RAJA CUDA dot product...\n";

  RAJA::ReduceSum<RAJA::cuda_reduce<CUDA_BLOCK_SIZE>, double> cudot(0.0);

  RAJA::forall<RAJA::cuda_exec<CUDA_BLOCK_SIZE>>(RAJA::RangeSegment(0, N), 
    [=] RAJA_DEVICE (int i) { 
    cudot += A[i] * B[i]; 
  });    

  dot = cudot.get();

  checkSolution(dot, N);
#endif

  memoryManager::deallocate(A);
  memoryManager::deallocate(B);

  std::cout << "\n DONE!...\n";

  return 0;
}
