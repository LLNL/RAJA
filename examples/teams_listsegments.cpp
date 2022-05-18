//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

#include "camp/resource.hpp"

/*
 *  Index sets and Segments Example
 *
 *  This example uses the daxpy kernel from a previous example. It
 *  illustrates how to use RAJA index sets and segments. This is 
 *  important for applications and algorithms that need to use 
 *  indirection arrays for irregular access. Combining range and
 *  list segments in a single index set, when possible, can 
 *  increase performance by allowing compilers to optimize for
 *  specific segment types (e.g., SIMD for range segments).
 *
 *  RAJA features shown:
 *    - `forall` loop iteration template method
 *    -  Index range segment 
 *    -  Index list segment 
 *    -  Strided index range segment 
 *    -  TypedIndexSet segment container
 *    -  Hierarchical execution policies
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

//----------------------------------------------------------------------------//
// Define types for ListSegments and indices used in examples
//----------------------------------------------------------------------------//
// _raja_list_segment_type_start
using IdxType = RAJA::Index_type;
using ListSegType = RAJA::TypedListSegment<IdxType>;
// _raja_list_segment_type_end

//
// Functions to check and print results
//
void checkResult(double* v1, double* v2, IdxType len);
void printResult(double* v, IdxType len);
 
int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA index sets and segments example...\n";

//
// Define vector length
//
  const IdxType N = 5;

//
// Allocate and initialize vector data
//
  double* a0 = memoryManager::allocate<double>(N);
  double* aref = memoryManager::allocate<double>(N);

  double* a = memoryManager::allocate<double>(N);
  double* b = memoryManager::allocate<double>(N);
  
  double c = 3.14159;
  
  for (IdxType i = 0; i < N; i++) {
    a0[i] = 1.0;
    b[i] = 2.0;
  }


//----------------------------------------------------------------------------//
//
// C-version of the daxpy kernel to set the reference result.
//
  std::cout << "\n Running C-version of daxpy to set reference result...\n";

  std::memcpy( aref, a0, N * sizeof(double) );

  for (IdxType i = 0; i < N; i++) {
    aref[i] += b[i] * c;
  }

  printResult(aref, N);

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)

//
// We create a new resource object and index set so that list segment 
// indices live in CUDA deviec memory.
//
  camp::resources::Resource cuda_res{camp::resources::Cuda()};

  std::vector<IdxType> idx1;
  for (IdxType i = 0; i < N; ++i) {
    idx1.push_back(i);
  }

  ListSegType idx1_list_cuda( &idx1[0], idx1.size(), cuda_res );

  std::memcpy( a, a0, N * sizeof(double) );

  using launch_policy = RAJA::expt::LaunchPolicy< RAJA::expt::cuda_launch_t<false>>;
  using loop_policy = RAJA::expt::LoopPolicy<RAJA::expt::cuda_global_thread_x>;

  //Use wrapper inside RAJA launch method
  RAJA::ListSegmentWrapper<IdxType> teams_lseg = idx1_list_cuda.MakeListSegmentWrapper();

  RAJA::expt::launch<launch_policy>
    (RAJA::expt::Grid(RAJA::expt::Teams(1), RAJA::expt::Threads(N)),
 
    [=] RAJA_HOST_DEVICE (RAJA::expt::LaunchContext ctx) {

      //RAJA::expt::loop<loop_policy>(ctx, idx1_list_cuda, [&] (int i) {
      RAJA::expt::loop<loop_policy>(ctx, teams_lseg, [&] (int i) { //use the wrapper class inside the kernel
          
          a[i] += b[i] * c;
          
        });
  });
  cudaDeviceSynchronize();

  checkResult(a, aref, N);
  printResult(a, N);
#endif

//----------------------------------------------------------------------------//

//
// Clean up. 
//
  memoryManager::deallocate(a); 
  memoryManager::deallocate(b); 
  memoryManager::deallocate(a0); 
  memoryManager::deallocate(aref); 
 
  std::cout << "\n DONE!...\n";
 
  return 0;
}

//
// Function to check result and report P/F.
//
void checkResult(double* v1, double* v2, IdxType len) 
{
  bool match = true;
  for (IdxType i = 0; i < len; i++) {
    if ( v1[i] != v2[i] ) { match = false; }
  }
  if ( match ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  } 
}

//
// Function to print result. 
//
void printResult(double* v, IdxType len) 
{
  std::cout << std::endl;
  for (IdxType i = 0; i < len; i++) {
    std::cout << "result[" << i << "] = " << v[i] << std::endl;
  }
  std::cout << std::endl;
} 

