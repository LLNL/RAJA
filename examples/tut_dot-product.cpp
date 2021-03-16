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

/*
 *  Vector Dot Product Example
 *
 *  Computes dot = (a,b), where a, b are vectors of 
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

#if defined(RAJA_ENABLE_HIP)
const int HIP_BLOCK_SIZE = 256;
#endif

//
//  Function to check dot product result.
//
void checkResult(double compdot, double refdot);

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA vector dot product example...\n";

//
// Define vector length
//
  const int N = 1000000;

//
// Allocate and initialize vector data
//
  double *a = memoryManager::allocate<double>(N);
  double *b = memoryManager::allocate<double>(N);

  for (int i = 0; i < N; ++i) {
    a[i] = 1.0;
    b[i] = 1.0;
  }

//----------------------------------------------------------------------------//

//
// C-style dot product operation.
//
  std::cout << "\n Running C-version of dot product...\n";

  // _csytle_dotprod_start
  double dot = 0.0;

  for (int i = 0; i < N; ++i) {
    dot += a[i] * b[i];
  }
  // _csytle_dotprod_end

  std::cout << "\t (a, b) = " << dot << std::endl;

  double dot_ref = dot;

//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA sequential dot product...\n";

  // _rajaseq_dotprod_start
  RAJA::ReduceSum<RAJA::seq_reduce, double> seqdot(0.0);

  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, N), [=] (int i) { 
    seqdot += a[i] * b[i]; 
  });

  dot = seqdot.get();
  // _rajaseq_dotprod_end

  std::cout << "\t (a, b) = " << dot << std::endl;

  checkResult(dot, dot_ref);


//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)
  std::cout << "\n Running RAJA OpenMP dot product...\n";

  // _rajaomp_dotprod_start
  RAJA::ReduceSum<RAJA::omp_reduce, double> ompdot(0.0);

  RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(0, N), [=] (int i) { 
    ompdot += a[i] * b[i]; 
  });    

  dot = ompdot.get();
  // _rajaomp_dotprod_end

  std::cout << "\t (a, b) = " << dot << std::endl;

  checkResult(dot, dot_ref);
#endif


//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)
  std::cout << "\n Running RAJA CUDA dot product...\n";

  // _rajacuda_dotprod_start
  RAJA::ReduceSum<RAJA::cuda_reduce, double> cudot(0.0);

  RAJA::forall<RAJA::cuda_exec<CUDA_BLOCK_SIZE>>(RAJA::RangeSegment(0, N), 
    [=] RAJA_DEVICE (int i) { 
    cudot += a[i] * b[i]; 
  });    

  dot = cudot.get();
  // _rajacuda_dotprod_end

  std::cout << "\t (a, b) = " << dot << std::endl;

  checkResult(dot, dot_ref);
#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_HIP)
  std::cout << "\n Running RAJA HIP dot product...\n";

  int *d_a = memoryManager::allocate_gpu<int>(N);
  int *d_b = memoryManager::allocate_gpu<int>(N);

  hipErrchk(hipMemcpy( d_a, a, N * sizeof(int), hipMemcpyHostToDevice ));
  hipErrchk(hipMemcpy( d_b, b, N * sizeof(int), hipMemcpyHostToDevice ));

  // _rajahip_dotprod_start
  RAJA::ReduceSum<RAJA::hip_reduce, double> hpdot(0.0);

  RAJA::forall<RAJA::hip_exec<HIP_BLOCK_SIZE>>(RAJA::RangeSegment(0, N),
    [=] RAJA_DEVICE (int i) {
    hpdot += d_a[i] * d_b[i];
  });

  dot = hpdot.get();
  // _rajahip_dotprod_end

  std::cout << "\t (a, b) = " << dot << std::endl;

  checkResult(dot, dot_ref);

  memoryManager::deallocate_gpu(d_a);
  memoryManager::deallocate_gpu(d_b);
#endif

//----------------------------------------------------------------------------//

  memoryManager::deallocate(a);
  memoryManager::deallocate(b);

  std::cout << "\n DONE!...\n";

  return 0;
}

//
//  Function to check computed dot product and report P/F.
//
void checkResult(double compdot, double refdot)
{
  if ( compdot == refdot ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
}

