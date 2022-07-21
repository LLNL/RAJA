//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
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

#if defined(RAJA_ENABLE_HIP)
const int HIP_BLOCK_SIZE = 256;
#endif

#if defined(RAJA_ENABLE_SYCL)
const int SYCL_BLOCK_SIZE = 256;
#endif

//
// Functions for checking and printing results
//
void checkResult(int* res, int len); 
void printResult(int* res, int len);


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA vector addition example...\n";

#if defined(RAJA_ENABLE_SYCL)
  memoryManager::sycl_res = new camp::resources::Resource{camp::resources::Sycl()};
  ::RAJA::sycl::detail::setQueue(memoryManager::sycl_res);
#endif

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

  std::cout << "\n Running C-style vector addition...\n";

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

  const bool Asynchronous = false;
  std::cout << "\n Running RAJA CUDA explicit (2 blocks per SM) vector addition...\n";

  RAJA::forall<RAJA::cuda_exec_explicit<CUDA_BLOCK_SIZE, 2, Asynchronous>>(RAJA::RangeSegment(0, N), 
    [=] RAJA_DEVICE (int i) { 
    c[i] = a[i] + b[i]; 
  });    

  checkResult(c, N);
//printResult(c, N);
#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_HIP)
  std::cout << "\n Running RAJA HIP vector addition...\n";

  int *d_a = memoryManager::allocate_gpu<int>(N);
  int *d_b = memoryManager::allocate_gpu<int>(N);
  int *d_c = memoryManager::allocate_gpu<int>(N);

  hipErrchk(hipMemcpy( d_a, a, N * sizeof(int), hipMemcpyHostToDevice ));
  hipErrchk(hipMemcpy( d_b, b, N * sizeof(int), hipMemcpyHostToDevice ));

  RAJA::forall<RAJA::hip_exec<HIP_BLOCK_SIZE>>(RAJA::RangeSegment(0, N),
    [=] RAJA_DEVICE (int i) {
    d_c[i] = d_a[i] + d_b[i];
  });

  hipErrchk(hipMemcpy( c, d_c, N * sizeof(int), hipMemcpyDeviceToHost ));

  checkResult(c, N);
//printResult(c, N);

  memoryManager::deallocate_gpu(d_a);
  memoryManager::deallocate_gpu(d_b);
  memoryManager::deallocate_gpu(d_c);
#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_SYCL)
  std::cout << "\n Running RAJA SYCL vector addition...\n";

  int *d_a = memoryManager::allocate_gpu<int>(N);
  int *d_b = memoryManager::allocate_gpu<int>(N);
  int *d_c = memoryManager::allocate_gpu<int>(N);

  memoryManager::sycl_res->memcpy(d_a, a, N * sizeof(int));
  memoryManager::sycl_res->memcpy(d_b, b, N * sizeof(int));

  RAJA::forall<RAJA::sycl_exec<SYCL_BLOCK_SIZE>>(RAJA::RangeSegment(0, N),
    [=] RAJA_DEVICE (int i) {
    d_c[i] = d_a[i] + d_b[i];
  });

  memoryManager::sycl_res->memcpy(c, d_c, N * sizeof(int));

  checkResult(c, N);
//printResult(c, N);

  memoryManager::deallocate_gpu(d_a);
  memoryManager::deallocate_gpu(d_b);
  memoryManager::deallocate_gpu(d_c);
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

