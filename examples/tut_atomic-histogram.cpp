//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cstring>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

/*
 *  Atomic Histogram Example
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

#if defined(RAJA_ENABLE_HIP)
const int HIP_BLOCK_SIZE = 256;
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
  // _range_atomic_histogram_start 
  RAJA::TypedRangeSegment<int> array_range(0, N);
  // _range_atomic_histogram_end 

  int* array = memoryManager::allocate<int>(N);
  int* bins = memoryManager::allocate<int>(M);

  RAJA::forall<RAJA::seq_exec>(array_range, [=](int i) {
                               
      array[i] = rand() % M;
      
  });
  //----------------------------------------------------------------------------//

  std::cout << "\n\n Running RAJA sequential binning" << std::endl;
  std::memset(bins, 0, M * sizeof(int));

  RAJA::forall<RAJA::seq_exec>(array_range, [=](int i) {
                                                      
    RAJA::atomicAdd<RAJA::seq_atomic>(&bins[array[i]], 1);

  });

  printBins(bins, M);

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)

  std::cout << "\n\n Running RAJA OMP binning" << std::endl;
  std::memset(bins, 0, M * sizeof(int));

  // _rajaomp_atomic_histogram_start 
  RAJA::forall<RAJA::omp_parallel_for_exec>(array_range, [=](int i) {
                          
    RAJA::atomicAdd<RAJA::omp_atomic>(&bins[array[i]], 1);
                                           
  });
  // _rajaomp_atomic_histogram_end

  printBins(bins, M);

//----------------------------------------------------------------------------//

  std::cout << "\n\n Running RAJA OMP binning with auto atomic" << std::endl;
  std::memset(bins, 0, M * sizeof(int));

  RAJA::forall<RAJA::omp_parallel_for_exec>(array_range, [=](int i) {
  
    RAJA::atomicAdd<RAJA::auto_atomic>(&bins[array[i]], 1);
  
  });

  printBins(bins, M);

#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)

  std::cout << "\n\nRunning RAJA CUDA binning" << std::endl;
  std::memset(bins, 0, M * sizeof(int));

  // _rajacuda_atomic_histogram_start 
  RAJA::forall< RAJA::cuda_exec<CUDA_BLOCK_SIZE> >(array_range, 
    [=] RAJA_DEVICE(int i) {
                          
    RAJA::atomicAdd<RAJA::cuda_atomic>(&bins[array[i]], 1);
                                                 
  });
  // _rajacuda_atomic_histogram_end

  printBins(bins, M);

//----------------------------------------------------------------------------//

  std::cout << "\n\nRunning RAJA CUDA binning with auto atomic" << std::endl;
  std::memset(bins, 0, M * sizeof(int));

  // _rajacuda_atomicauto_histogram_start 
  RAJA::forall< RAJA::cuda_exec<CUDA_BLOCK_SIZE> >(array_range, 
    [=] RAJA_DEVICE(int i) {

    RAJA::atomicAdd<RAJA::auto_atomic>(&bins[array[i]], 1);

  });
  // _rajacuda_atomicauto_histogram_end

  printBins(bins, M);
  
#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_HIP)

  std::cout << "\n\nRunning RAJA HIP binning" << std::endl;
  std::memset(bins, 0, M * sizeof(int));

  int* d_array = memoryManager::allocate_gpu<int>(N);
  int* d_bins  = memoryManager::allocate_gpu<int>(M);
  hipErrchk(hipMemcpy( d_array, array, N * sizeof(int), hipMemcpyHostToDevice ));
  hipErrchk(hipMemcpy( d_bins, bins, M * sizeof(int), hipMemcpyHostToDevice ));

  // _rajahip_atomic_histogram_start 
  RAJA::forall< RAJA::hip_exec<HIP_BLOCK_SIZE> >(array_range, 
    [=] RAJA_DEVICE(int i) {

    RAJA::atomicAdd<RAJA::hip_atomic>(&d_bins[d_array[i]], 1);

  });
  // _rajahip_atomic_histogram_end

  hipErrchk(hipMemcpy( bins, d_bins, M * sizeof(int), hipMemcpyDeviceToHost ));

  printBins(bins, M);

//----------------------------------------------------------------------------//

  std::cout << "\n\nRunning RAJA HIP binning with auto atomic" << std::endl;
  std::memset(bins, 0, M * sizeof(int));
  hipErrchk(hipMemcpy( d_bins, bins, M * sizeof(int), hipMemcpyHostToDevice ));

  // _rajahip_atomicauto_histogram_start
  RAJA::forall< RAJA::hip_exec<HIP_BLOCK_SIZE> >(array_range, 
    [=] RAJA_DEVICE(int i) {

    RAJA::atomicAdd<RAJA::auto_atomic>(&d_bins[d_array[i]], 1);

  });
  // _rajahip_atomicauto_histogram_end

  hipErrchk(hipMemcpy( bins, d_bins, M * sizeof(int), hipMemcpyDeviceToHost ));

  printBins(bins, M);

  memoryManager::deallocate_gpu(d_array);
  memoryManager::deallocate_gpu(d_bins);
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
