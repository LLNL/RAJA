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
#include <algorithm>
#include <numeric>
#include <random>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

/*
 *  Scan Example
 *
 *  Example shows how to perform RAJA inclusive and exclusive scan operations
 *  for integer arrays, including in-place, using different operators. 
 *  Other array data types, operators, etc. are similar
 *
 *  RAJA features shown:
 *    - `RAJA::inclusive_scan` and `RAJA::inclusive_scan_inplace` methods
 *    - `RAJA::exclusive_scan` and `RAJA::exclusive_scan_inplace` methods
 *    -  RAJA operators
 *    -  Execution policies
 *
 * If CUDA is enabled, CUDA unified memory is used.
 */

/*
  CUDA_BLOCK_SIZE - specifies the number of threads in a CUDA thread block
*/
#if defined(RAJA_ENABLE_CUDA)
const int CUDA_BLOCK_SIZE = 16;
#endif

//
// Functions for checking results and printing vectors
//
template <typename Function, typename T>
void checkInclusiveScanResult(const T* in, const T* out, int N);
//
template <typename Function, typename T>
void checkExclusiveScanResult(const T* in, const T* out, int N);
//
template <typename T>
void printArray(const T* v, int N);


int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA scan example...\n";

//
// Define array length
//
  const int N = 20;

//
// Allocate and array vector data
//
  int* in = memoryManager::allocate<int>(N); 
  int* out = memoryManager::allocate<int>(N); 

  std::iota(in, in+N, -1);

  std::shuffle(in, in + N, std::mt19937{std::random_device{}()});
  std::cout << "\n in values...\n";
  printArray(in, N);  
  std::cout << "\n";


//----------------------------------------------------------------------------//
// Perform various sequential scans to illustrate inclusive/exclusive,
// in-place, default scans with different operators
//----------------------------------------------------------------------------//

  std::cout << "\n Running sequential inclusive_scan (default)...\n";

  RAJA::inclusive_scan<RAJA::seq_exec>(in, in + N, out);

  checkInclusiveScanResult<RAJA::operators::plus<int>>(in, out, N);
  printArray(out, N);  
  std::cout << "\n";

//----------------------------------------------------------------------------//

  std::cout << "\n Running sequential inclusive_scan (plus)...\n";

  RAJA::inclusive_scan<RAJA::seq_exec>(in, in + N, out,
                                       RAJA::operators::plus<int>{});

  checkInclusiveScanResult<RAJA::operators::plus<int>>(in, out, N);
  printArray(out, N);  
  std::cout << "\n";

//----------------------------------------------------------------------------//

  std::cout << "\n Running sequential exclusive_scan (plus)...\n";

  RAJA::exclusive_scan<RAJA::seq_exec>(in, in + N, out,
                                       RAJA::operators::plus<int>{});

  checkExclusiveScanResult<RAJA::operators::plus<int>>(in, out, N);
  printArray(out, N);  
  std::cout << "\n";

//----------------------------------------------------------------------------//

  std::cout << "\n Running sequential inclusive_scan_inplace (minimum)...\n";

  std::copy_n(in, N, out);

  RAJA::inclusive_scan_inplace<RAJA::seq_exec>(out, out + N,
                                               RAJA::operators::minimum<int>{});

  checkInclusiveScanResult<RAJA::operators::minimum<int>>(in, out, N);
  printArray(out, N);
  std::cout << "\n";

//----------------------------------------------------------------------------//

  std::cout << "\n Running sequential exclusive_scan_inplace (maximum)...\n";

  std::copy_n(in, N, out);

  RAJA::exclusive_scan_inplace<RAJA::seq_exec>(out, out + N,
                                       RAJA::operators::maximum<int>{});

  checkExclusiveScanResult<RAJA::operators::maximum<int>>(in, out, N);
  printArray(out, N);
  std::cout << "\n";


#if defined(RAJA_ENABLE_OPENMP)

//----------------------------------------------------------------------------//
// Perform a couple of OpenMP scans...
//----------------------------------------------------------------------------//

  std::cout << "\n Running OpenMP inclusive_scan (plus)...\n";

  RAJA::inclusive_scan<RAJA::omp_parallel_for_exec>(in, in + N, out,
                                                    RAJA::operators::plus<int>{});

  checkInclusiveScanResult<RAJA::operators::plus<int>>(in, out, N);
  printArray(out, N);
  std::cout << "\n";

//----------------------------------------------------------------------------//

  std::cout << "\n Running OpenMP exclusive_scan_inplace (plus)...\n";

  std::copy_n(in, N, out);

  RAJA::exclusive_scan_inplace<RAJA::omp_parallel_for_exec>(out, out + N,
                                                            RAJA::operators::plus<int>{});

  checkExclusiveScanResult<RAJA::operators::plus<int>>(in, out, N);
  printArray(out, N);
  std::cout << "\n";

#endif


#if defined(RAJA_ENABLE_CUDA)

//----------------------------------------------------------------------------//
// Perform a couple of CUDA scans...
//----------------------------------------------------------------------------//

  std::cout << "\n Running CUDA inclusive_scan_inplace (plus)...\n";

  std::copy_n(in, N, out);

  RAJA::inclusive_scan_inplace<RAJA::cuda_exec<CUDA_BLOCK_SIZE>>(out, out + N,
                                       RAJA::operators::plus<int>{});

  checkInclusiveScanResult<RAJA::operators::plus<int>>(in, out, N);
  printArray(out, N);
  std::cout << "\n";

//----------------------------------------------------------------------------//

  std::cout << "\n Running CUDA exclusive_scan (plus)...\n";
  RAJA::exclusive_scan<RAJA::cuda_exec<CUDA_BLOCK_SIZE>>(in, in + N, out,
                                       RAJA::operators::plus<int>{});

  checkExclusiveScanResult<RAJA::operators::plus<int>>(in, out, N);
  printArray(out, N);
  std::cout << "\n";

#endif

//----------------------------------------------------------------------------//

//
// Clean up.
//
  memoryManager::deallocate(in);
  memoryManager::deallocate(out);

  std::cout << "\n DONE!...\n";

  return 0;
}


//
// Function to check inclusive scan result
//
template <typename Function, typename T>
void checkInclusiveScanResult(const T* in, const T* out, int N)
{
  T val = Function::identity();
  for (int i = 0; i < N; ++i) {
    val = Function()(val, in[i]);
    if (out[i] != val) {
      std::cout << "\n\t result -- WRONG\n";
      std::cout << "\t" << out[i] << " != " << val 
                << " (at index " << i << ")\n";
    }
  }
  std::cout << "\n\t result -- CORRECT\n";
}

//
// Function to check exclusive scan result
//
template <typename Function, typename T>
void checkExclusiveScanResult(const T* in, const T* out, int N)
{
  T val = Function::identity();
  for (int i = 0; i < N; ++i) {
    if (out[i] != val) {
      std::cout << "\n\t result -- WRONG\n";
      std::cout << "\t" << out[i] << " != " << val 
                << " (at index " << i << ")\n";
    }
    val = Function()(val, in[i]);
  }
  std::cout << "\n\t result -- CORRECT\n";
}

//
// Function to print vector.
//
template <typename T>
void printArray(const T* v, int N)
{
  std::cout << std::endl;
  for (int i = 0; i < N; ++i) { std::cout << " " << v[i]; }
  std::cout << std::endl;
}
