//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "RAJA/RAJA.hpp"
#include "RAJA/util/Timer.hpp"


#include "memoryManager.hpp"

/*
 *  RAJA Reshape method
 *
 *  This example will intialize array using
 *  the RAJA Reshape method. The Reshape
 *  method offers right and left most unit
 *  stride.
 *
 */

/*
 * Define number of threads in a GPU thread block
 */
#if defined(RAJA_ENABLE_CUDA)
constexpr int CUDA_BLOCK_SIZE = 256;
#endif

#if defined(RAJA_ENABLE_HIP)
constexpr int HIP_BLOCK_SIZE = 256;
#endif

//
//Function for checking results
//
void checkResult(int *ptr, int K, int N, int M);

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA reshape method example....\n"<< std::endl;

  const int K = 3;
  const int N = 1;
  const int M = 2;

  // Allocate memory for pointer
  int *Rptr = memoryManager::allocate<int>(K * N * M);
  int *Lptr = memoryManager::allocate<int>(K * N * M);

//----------------------------------------------------------------------------//
//
// Initialize memory using right most unit stride
//
//----------------------------------------------------------------------------//
  std::cout << "\n\nInitialize array with right most indexing...\n";
  auto Rview = RAJA::Reshape<RAJA::layout_right>::get(Rptr, K, N, M);

  for(int k = 0; k < K; ++k) {
    for(int n = 0; n < N; ++n) {
      for(int m = 0; m < M; ++m) {
        const int idx = m + M * (n + N * k);
        Rview(k,n,m) = idx;
      }
    }
  }

  checkResult(Rptr, K, N, M);


//----------------------------------------------------------------------------//
//
// Initialize memory using left most unit stride
//
//----------------------------------------------------------------------------//
  std::cout << "\n\nInitialize array with left most indexing...\n";

  auto Lview = RAJA::Reshape<RAJA::layout_left>::get(Lptr, K, N, M);

  //Note the loop ordering has change from above
  for(int m = 0; m < M; ++m) {
    for(int n = 0; n < N; ++n) {
      for(int k = 0; k < K; ++k) {

        const int idx = k + K * (n + N * m);
        Lview(k,m,n) = idx;
      }
    }
  }

  checkResult(Lptr, K, N, M);

//
// Clean up.
//
  memoryManager::deallocate(Rptr);
  memoryManager::deallocate(Lptr);

  std::cout << "\n DONE!...\n";
  return 0;
}

//
// check result
//
void checkResult(int *ptr, int K, int N, int M)
{

  bool status = true;

  for(int k = 0; k < K; ++k) {
    for(int n = 0; n < N; ++n) {
      for(int m = 0; m < M; ++m) {
        const int idx = m + M * (n + N * k);
        if (std::abs(ptr[idx] - idx) > 0) {
          status = false;
        }
      }
    }
  }

  if ( status ) {
    std::cout << "\tresult -- PASS\n";
  } else {
    std::cout << "\tresult -- FAIL\n";
  }
}
