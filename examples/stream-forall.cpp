//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"
#include "RAJA/util/resource.hpp"

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
  const int N = 100000;

//
// Allocate and initialize vector data
//
  int *a = memoryManager::allocate<int>(N);
  int *b = memoryManager::allocate<int>(N);
  int *c = memoryManager::allocate<int>(N);

  int *a_ = memoryManager::allocate<int>(N);
  int *b_ = memoryManager::allocate<int>(N);
  int *c_ = memoryManager::allocate<int>(N);


  for (int i = 0; i < N; ++i) {
    a[i] = -i;
    b[i] = i;
    a_[i] = -i;
    b_[i] = i;

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

  RAJA::resources::Resource host{RAJA::resources::Host()};

  // _rajaseq_vector_add_start
  RAJA::forall<RAJA::seq_exec>(host, RAJA::RangeSegment(0, N), [=] (int i) { 
    c[i] = a[i] + b[i]; 
  });
  // _rajaseq_vector_add_end

  checkResult(c, N);
//printResult(c, N);

//----------------------------------------------------------------------------//
// RAJA::loop_exec policy enforces strictly sequential execution.... 
//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA loop vector addition...\n";

  // _rajaseq_vector_add_start
  RAJA::forall<RAJA::loop_exec>(host, RAJA::RangeSegment(0, N), [=] (int i) { 
    c[i] = a[i] + b[i]; 
  });
  // _rajaseq_vector_add_end

  checkResult(c, N);
//printResult(c, N);

//----------------------------------------------------------------------------//
// RAJA::omp_for_parallel_exec policy enforces strictly sequential execution.... 
//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA omp_parallel<seq_exec> vector addition...\n";

  // _rajaseq_vector_add_start
  RAJA::forall<RAJA::omp_parallel_exec<RAJA::seq_exec>>(host, RAJA::RangeSegment(0, N), [=] RAJA_DEVICE (int i) { 
    c[i] = a[i] + b[i]; 
  });
  // _rajaseq_vector_add_end

  checkResult(c, N);
//printResult(c, N);

//----------------------------------------------------------------------------//
// RAJA::omp_for_nowait_exec policy enforces strictly sequential execution.... 
//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA omp_for_nowait vector addition...\n";

  // _rajaseq_vector_add_start
  RAJA::forall<RAJA::omp_for_nowait_exec>(host, RAJA::RangeSegment(0, N), [=] (int i) { 
    c[i] = a[i] + b[i]; 
  });
  // _rajaseq_vector_add_end

  checkResult(c, N);
//printResult(c, N);

//----------------------------------------------------------------------------//
// RAJA::omp_for_exec policy enforces strictly sequential execution.... 
//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA omp_for_exec vector addition...\n";

  // _rajaseq_vector_add_start
  RAJA::forall<RAJA::omp_for_exec>(host, RAJA::RangeSegment(0, N), [=] (int i) { 
    c[i] = a[i] + b[i]; 
  });
  // _rajaseq_vector_add_end

  checkResult(c, N);
//printResult(c, N);

//----------------------------------------------------------------------------//
// RAJA::omp_for_static policy enforces strictly sequential execution.... 
//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA omp_for_static vector addition...\n";

  // _rajaseq_vector_add_start
  RAJA::forall<RAJA::omp_for_static<8>>(host, RAJA::RangeSegment(0, N), [=] RAJA_HOST (int i) { 
    c[i] = a[i] + b[i]; 
  });
  // _rajaseq_vector_add_end

  checkResult(c, N);
//printResult(c, N);

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)
  std::cout << "\n Running RAJA CUDA vector addition...\n";
  RAJA::resources::Resource res_cuda1{RAJA::resources::Cuda()};
  RAJA::resources::Resource res_cuda2{RAJA::resources::Cuda()};

  // _rajacuda_vector_add_start
  auto e = RAJA::forall<RAJA::cuda_exec<CUDA_BLOCK_SIZE, true>>(res_cuda1, RAJA::RangeSegment(0, N), 
    [=] RAJA_DEVICE (int i) { 
    c[i] = a[i] + b[i]; 
  });    

  auto e_ = RAJA::forall<RAJA::cuda_exec<CUDA_BLOCK_SIZE, true>>(res_cuda2, RAJA::RangeSegment(0, N), 
    [=] RAJA_DEVICE (int i) { 
    c_[i] = a_[i] + b_[i]; 
  }); 
  // _rajacuda_vector_add_end

  checkResult(c, N);
  checkResult(c_, N);
//printResult(c, N);
#endif

////----------------------------------------------------------------------------//
//
//#if defined(RAJA_ENABLE_HIP)
//  std::cout << "\n Running RAJA HIP vector addition...\n";
//
//  int *d_a = memoryManager::allocate_gpu<int>(N);
//  int *d_b = memoryManager::allocate_gpu<int>(N);
//  int *d_c = memoryManager::allocate_gpu<int>(N);
//
//  hipErrchk(hipMemcpy( d_a, a, N * sizeof(int), hipMemcpyHostToDevice ));
//  hipErrchk(hipMemcpy( d_b, b, N * sizeof(int), hipMemcpyHostToDevice ));
//
//  RAJA::forall<RAJA::hip_exec<HIP_BLOCK_SIZE>>(RAJA::RangeSegment(0, N),
//    [=] RAJA_DEVICE (int i) {
//    d_c[i] = d_a[i] + d_b[i];
//  });
//
//  hipErrchk(hipMemcpy( c, d_c, N * sizeof(int), hipMemcpyDeviceToHost ));
//
//  checkResult(c, N);
////printResult(c, N);
//
//  memoryManager::deallocate_gpu(d_a);
//  memoryManager::deallocate_gpu(d_b);
//  memoryManager::deallocate_gpu(d_c);
//#endif
//
////----------------------------------------------------------------------------//

//
// Clean up.
//
  memoryManager::deallocate(a);
  memoryManager::deallocate(b);
  memoryManager::deallocate(c);
  memoryManager::deallocate(a_);
  memoryManager::deallocate(b_);
  memoryManager::deallocate(c_);

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

