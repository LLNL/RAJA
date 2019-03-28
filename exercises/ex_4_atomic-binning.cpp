//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
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
  RAJA::TypedRangeSegment<int> array_range(0, N);

  int* array = memoryManager::allocate<int>(N);
  int* bins = memoryManager::allocate<int>(M);

  RAJA::forall<RAJA::seq_exec>(array_range, [=](int i) {
                               
      array[i] = rand() % M;
      
  });
  //----------------------------------------------------------------------------//

  std::cout << "\n\n Running RAJA sequential binning" << std::endl;
  std::memset(bins, 0, M * sizeof(int));

  //TODO: RAJA sequential variant using atomics

  printBins(bins, M);

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)

  std::cout << "\n\n Running RAJA OMP binning" << std::endl;
  std::memset(bins, 0, M * sizeof(int));

  //TODO: RAJA OpenMP variant using OpenMP or auto atomics

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
