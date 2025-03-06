//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <iostream>

#include "RAJA/RAJA.hpp"
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
  int *Cptr = memoryManager::allocate<int>(K * N * M);

//----------------------------------------------------------------------------//
//
// Initialize memory using right most unit stride
//
//----------------------------------------------------------------------------//
  std::cout << "\n\nInitialize array with right most indexing...\n";
  // _Reshape_right_start
  auto Rview = RAJA::Reshape<RAJA::layout_right>::get(Rptr, K, N, M);
  // _Reshape_right_end

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
  // _Reshape_left_start
  auto Lview = RAJA::Reshape<RAJA::layout_left>::get(Lptr, K, N, M);
  // _Reshape_left_end

  //Note the loop ordering has change from above
  for(int m = 0; m < M; ++m) {
    for(int n = 0; n < N; ++n) {
      for(int k = 0; k < K; ++k) {

        const int idx = k + K * (n + N * m);
        Lview(k,n,m) = idx;
      }
    }
  }

  checkResult(Lptr, K, N, M);


//----------------------------------------------------------------------------//
//
// Initialize memory using custom ordering, longest stride starts at the left,
// right most is assumed to have unit stride.
//
//----------------------------------------------------------------------------//
  std::cout << "\n\nInitialize array with custom indexing...\n";

  // _Reshape_custom_start
  using custom_seq = std::index_sequence<2U,0U,1U>;

  auto Cview = RAJA::Reshape<custom_seq>::get(Cptr, K, N, M);
  // _Reshape_custom_end

  //Note the loop ordering has change from above
  for(int m = 0; m < M; ++m) {
    for(int k = 0; k < K; ++k) {
      for(int n = 0; n < N; ++n) {

        const int idx = n + N * (k + K * m);
        Cview(k,n,m) = idx;
      }
    }
  }

  checkResult(Cptr, K, N, M);


//
// Clean up.
//
  memoryManager::deallocate(Rptr);
  memoryManager::deallocate(Lptr);
  memoryManager::deallocate(Cptr);

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
        if (ptr[idx] != idx) {
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
