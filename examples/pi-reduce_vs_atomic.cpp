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
#include <iostream>
#include <iomanip>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

/*
 *  Pi Example
 *
 *  Computes an approximation to pi. It illustrates that RAJA reduction
 *  and atomic operations are used similarly for different progamming
 *  model backends.
 *
 *  RAJA features shown:
 *    - `forall` loop iteration template method
 *    - Index range segment
 *    - Sum reduction
 *    - Atomic add
 *
 *  If CUDA is enabled, CUDA unified memory is used.
 */

/*
  CUDA_BLOCK_SIZE - specifies the number of threads in a CUDA thread block
*/
#if defined(RAJA_ENABLE_CUDA)
const int CUDA_BLOCK_SIZE = 256;
#endif

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA pi example...\n";

//
// Define RangeSegment to enumerate "bins" used in pi approximation,
// and memory location for atomic add operation.
//
  const int num_bins = 512 * 512;
  RAJA::RangeSegment bins(0, num_bins); 

  double* atomic_pi = memoryManager::allocate<double>(1);

// Set precision for printing pi
  int prec = 16;


//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA sequential pi approximation (reduction)...\n";

  using EXEC_POL1   = RAJA::seq_exec;
  using REDUCE_POL1 = RAJA::seq_reduce; 

  RAJA::ReduceSum<REDUCE_POL1, double> seq_pi(0.0);

  RAJA::forall<EXEC_POL1>(bins, [=](int i) {
      double x = (double(i) + 0.5) / num_bins;
      seq_pi += 4.0 / (1.0 + x * x);
  });

  std::cout << "\tpi = " << std::setprecision(prec) 
            << seq_pi.get() / num_bins << std::endl;


  std::cout << "\n Running RAJA sequential pi approximation (atomic)...\n";

  *atomic_pi = 0;

  using ATOMIC_POL1 = RAJA::atomic::seq_atomic;

  RAJA::forall<EXEC_POL1>(bins, [=](int i) {
      double x = (double(i) + 0.5) / num_bins;
      RAJA::atomic::atomicAdd<ATOMIC_POL1>(atomic_pi, 4.0 / (1.0 + x * x));
  });

  std::cout << "\tpi = " << std::setprecision(prec) 
            << (*atomic_pi) / num_bins << std::endl;


//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)

  std::cout << "\n Running RAJA OpenMP pi approximation (reduction)...\n";

  using EXEC_POL2   = RAJA::omp_parallel_for_exec;
  using REDUCE_POL2 = RAJA::omp_reduce;

  RAJA::ReduceSum<REDUCE_POL2, double> omp_pi(0.0);

  RAJA::forall<EXEC_POL2>(bins, [=](int i) {
      double x = (double(i) + 0.5) / num_bins;
      omp_pi += 4.0 / (1.0 + x * x);
  });

  std::cout << "\tpi = " << std::setprecision(prec)
            << omp_pi.get() / num_bins << std::endl;


  std::cout << "\n Running RAJA OpenMP pi approximation (atomic)...\n";

  *atomic_pi = 0;

  using ATOMIC_POL2 = RAJA::atomic::omp_atomic;

  RAJA::forall<EXEC_POL2>(bins, [=](int i) {
      double x = (double(i) + 0.5) / num_bins;
      RAJA::atomic::atomicAdd<ATOMIC_POL2>(atomic_pi, 4.0 / (1.0 + x * x));
  });

  std::cout << "\tpi = " << std::setprecision(prec)
            << (*atomic_pi) / num_bins << std::endl;

#endif


//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)

  std::cout << "\n Running RAJA CUDA pi approximation (reduction)...\n";

  using EXEC_POL3   = RAJA::cuda_exec<CUDA_BLOCK_SIZE>;
  using REDUCE_POL3 = RAJA::cuda_reduce<CUDA_BLOCK_SIZE>;

  RAJA::ReduceSum<REDUCE_POL3, double> cuda_pi(0.0);

  RAJA::forall<EXEC_POL3>(bins, [=] RAJA_DEVICE (int i) {
      double x = (double(i) + 0.5) / num_bins;
      cuda_pi += 4.0 / (1.0 + x * x);
  });

  std::cout << "\tpi = " << std::setprecision(prec)
            << cuda_pi.get() / num_bins << std::endl;


  std::cout << "\n Running RAJA CUDA pi approximation (atomic)...\n";

  *atomic_pi = 0;

  using ATOMIC_POL3 = RAJA::atomic::cuda_atomic;

  RAJA::forall<EXEC_POL3>(bins, [=] RAJA_DEVICE (int i) {
      double x = (double(i) + 0.5) / num_bins;
      RAJA::atomic::atomicAdd<ATOMIC_POL3>(atomic_pi, 4.0 / (1.0 + x * x));
  });

  std::cout << "\tpi = " << std::setprecision(prec)
            << (*atomic_pi) / num_bins << std::endl;

#endif

//----------------------------------------------------------------------------//

//
// Clean up.
//
  memoryManager::deallocate(atomic_pi);

  std::cout << "\n DONE!...\n";

  return 0;
}
