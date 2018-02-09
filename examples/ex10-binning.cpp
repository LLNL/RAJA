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
#include <cstring>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

/*
 *  Binning Example
 *
 *  Given an array of length N containing integers ranging from [0, M),
 *  this example uses RAJA atomics to count the number of instances a
 *  number between 0 and M appear.
 *
 *  RAJA features shown:
 *    - `forall` loop iteration template method
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

template <typename T>
void printBins(T* bins, int M);

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{
  //
  // Define the inital array containing values between 0 and M and
  // create the iteration bounds
  //
  int M = 10;
  int N = 30;
  RAJA::RangeSegment array_range(0, N);

  int* array = memoryManager::allocate<int>(N);
  int* bins = memoryManager::allocate<int>(M);

  RAJA::forall<RAJA::seq_exec>(array_range, [=](int i) {
                               
      array[i] = rand() % M;
      
    });
  //----------------------------------------------------------------------------//

  std::cout << "\n\n Running RAJA sequential binning" << std::endl;
  std::memset(bins, 0, M * sizeof(int));

  using EXEC_POL1 = RAJA::seq_exec;
  using ATOMIC_POL1 = RAJA::atomic::seq_atomic;

  RAJA::forall<EXEC_POL1>(array_range, [=](int i) {
                                                      
      RAJA::atomic::atomicAdd<ATOMIC_POL1>(&bins[array[i]], 1);

    });

  printBins(bins, M);

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)

  std::cout << "\n\n Running RAJA OMP binning" << std::endl;
  std::memset(bins, 0, M * sizeof(int));

  using EXEC_POL2 = RAJA::omp_parallel_for_exec;
  using ATOMIC_POL2 = RAJA::atomic::omp_atomic;

  RAJA::forall<EXEC_POL2>(array_range, [=](int i) {
                          
      RAJA::atomic::atomicAdd<ATOMIC_POL2>(&bins[array[i]], 1);
                                           
    });

  printBins(bins, M);

//----------------------------------------------------------------------------//

  std::cout << "\n\n Running RAJA OMP binning with auto atomic" << std::endl;
  std::memset(bins, 0, M * sizeof(int));

  using EXEC_POL2 = RAJA::omp_parallel_for_exec;
  using ATOMIC_POL3 = RAJA::atomic::auto_atomic;

  RAJA::forall<EXEC_POL2>(array_range, [=](int i) {
  
      RAJA::atomic::atomicAdd<ATOMIC_POL3>(&bins[array[i]], 1);
  
    });

  printBins(bins, M);

#endif
//----------------------------------------------------------------------------//


#if defined(RAJA_ENABLE_CUDA)

  std::cout << "\n\nRunning RAJA CUDA binning" << std::endl;
  std::memset(bins, 0, M * sizeof(int));

  using EXEC_POL4 = RAJA::cuda_exec<CUDA_BLOCK_SIZE>;
  using ATOMIC_POL4 = RAJA::atomic::cuda_atomic;

  RAJA::forall<EXEC_POL4>(array_range, [=] RAJA_DEVICE(int i) {
                          
      RAJA::atomic::atomicAdd<ATOMIC_POL4>(&bins[array[i]], 1);
                                                 
    });

  printBins(bins, M);

//----------------------------------------------------------------------------//

  std::cout << "\n\nRunning RAJA CUDA binning with auto atomic" << std::endl;
  std::memset(bins, 0, M * sizeof(int));

  using ATOMIC_POL5 = RAJA::atomic::auto_atomic;

  RAJA::forall<EXEC_POL4>(array_range, [=] RAJA_DEVICE(int i) {

      RAJA::atomic::atomicAdd<ATOMIC_POL5>(&bins[array[i]], 1);

    });

  printBins(bins, M);
  
#endif
  //----------------------------------------------------------------------------//


  //
  // Clean up dellacate data
  //
  memoryManager::deallocate(array);
  memoryManager::deallocate(bins);

  std::cout << "\n DONE!...\n";

  return 0;
}

template <typename T>
void printBins(T* bins, int M)
{

  std::cout << "Number of instances |";
  for (int i = 0; i < M; ++i) {
    std::cout << bins[i] << " ";
  }
  std::cout << "" << std::endl;

  std::cout << "---------------------------";
  for (int i = 0; i < M; ++i) {
    std::cout << "-"
              << "";
  }
  std::cout << "" << std::endl;

  std::cout << "Index id            |";
  for (int i = 0; i < M; ++i) {
    std::cout << i << " ";
  }
  std::cout << "\n" << std::endl;
}
