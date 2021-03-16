//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"
#include "camp/resource.hpp"

using namespace camp::resources;

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
  Resource hostRc{Host()};
  int *a = hostRc.allocate<int>(N);
  int *b = hostRc.allocate<int>(N);
  int *c = hostRc.allocate<int>(N);

  for (int i = 0; i < N; ++i) {
    a[i] = -i;
    b[i] = i;
  }


//----------------------------------------------------------------------------//

  std::cout << "\n Running C-style vector addition...\n";

  // _cstyle_vector_add_start
  for (int i = 0; i < N; ++i) {
    c[i] = a[i] + b[i];
  }
  // _cstyle_vector_add_end

  checkResult(c, N);
//printResult(c, N);


//----------------------------------------------------------------------------//
// RAJA::seq_exec policy enforces strictly sequential execution.... 
//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA sequential vector addition...\n";

  // _rajaseq_vector_add_start
  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, N), [=] (int i) { 
    c[i] = a[i] + b[i]; 
  });
  // _rajaseq_vector_add_end

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

  // _rajaomp_vector_add_start
  RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(0, N), [=] (int i) { 
    c[i] = a[i] + b[i]; 
  });
  // _rajaomp_vector_add_end

  checkResult(c, N);
//printResult(c, N);
#endif


//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)
  std::cout << "\n Running RAJA CUDA vector addition...\n";

  Resource cudaRc{Cuda()};
  int *d_a = cudaRc.allocate<int>(N);
  int *d_b = cudaRc.allocate<int>(N);
  int *d_c = cudaRc.allocate<int>(N);

  cudaRc.memcpy(d_a,a,N*sizeof(int));
  cudaRc.memcpy(d_b,b,N*sizeof(int));
  cudaRc.memcpy(d_c,c,N*sizeof(int));

  // _rajacuda_vector_add_start
  RAJA::forall<RAJA::cuda_exec<CUDA_BLOCK_SIZE>>(RAJA::RangeSegment(0, N), 
    [=] RAJA_DEVICE (int i) { 
    d_c[i] = d_a[i] + d_b[i];
  });    
  // _rajacuda_vector_add_end

  cudaRc.memcpy(c,d_c,N*sizeof(int));
  cudaRc.deallocate(d_a);
  cudaRc.deallocate(d_b);
  cudaRc.deallocate(d_c);

  checkResult(c, N);
//printResult(c, N);
#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_HIP)
  std::cout << "\n Running RAJA HIP vector addition...\n";

  Resource hipRc{Hip()};
  int *d_a = hipRc.allocate<int>(N);
  int *d_b = hipRc.allocate<int>(N);
  int *d_c = hipRc.allocate<int>(N);

  hipRc.memcpy(d_a,a,N*sizeof(int));
  hipRc.memcpy(d_b,b,N*sizeof(int));
  hipRc.memcpy(d_c,c,N*sizeof(int));

  // _rajahip_vector_add_start
  RAJA::forall<RAJA::hip_exec<HIP_BLOCK_SIZE>>(RAJA::RangeSegment(0, N),
    [=] RAJA_DEVICE (int i) {
    d_c[i] = d_a[i] + d_b[i];
  });
  // _rajahip_vector_add_end

  hipRc.memcpy(c,d_c,N*sizeof(int));
  hipRc.deallocate(d_a);
  hipRc.deallocate(d_b);
  hipRc.deallocate(d_c);

  checkResult(c, N);
//printResult(c, N);
#endif

//----------------------------------------------------------------------------//

//
// Clean up.
//

  hostRc.deallocate(a);
  hostRc.deallocate(b);
  hostRc.deallocate(c);

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

