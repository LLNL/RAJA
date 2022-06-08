//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

/*
 *  Vector Addition Example with dynamic policy selection
 *
 *  Computes c = a + b, where a, b, c are vectors of ints.
 *  It illustrates similarities between a  C-style for-loop and a RAJA 
 *  forall loop.
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

#if defined(RAJA_ENABLE_SYCL)
const int SYCL_BLOCK_SIZE = 256;
#endif

//
// Functions for checking and printing results
//
void checkResult(int* res, int len); 
void printResult(int* res, int len);

int main(int argc, char *argv[])
{

  if(argc != 2) {
    RAJA_ABORT_OR_THROW("Usage ./run-time-forall {host, host-parallel, device}");
  }

  //
  // Run time policy section is demonstrated in this example by specifying
  // kernel exection space as a command line argument (host, host-parallel, device).
  // Example usage ./teams_reductions host or ./teams_reductions device
  //
  std::string exec_space = argv[1];
  if(!(exec_space.compare("host") == 0 || exec_space.compare("host-parallel") == 0 || 
       exec_space.compare("device") == 0 )){
    RAJA_ABORT_OR_THROW("Usage ./teams_reductions host or ./teams_reductions device");
    return 0;
  }

  std::cout << "\n\nRAJA vector addition example...\n";

#if defined(RAJA_ENABLE_SYCL)
  memoryManager::sycl_res = new camp::resources::Resource{camp::resources::Sycl()};
  ::RAJA::sycl::detail::setQueue(memoryManager::sycl_res);
#endif

//
// Define vector length
//
  const int N = 1000000;

//
// Allocate and initialize vector data
//
  int *a = memoryManager::allocate<int>(N);
  int *b = memoryManager::allocate<int>(N);
  int *c = memoryManager::allocate<int>(N);

  for (int i = 0; i < N; ++i) {
    a[i] = -i;
    b[i] = i;
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
// Example of dynamic policy selection for forall
//----------------------------------------------------------------------------//
  RAJA::expt::exec_policy dynamic_policy;
  if(exec_space.compare("host") == 0)
    { dynamic_policy = RAJA::expt::host_seq; printf("Running RAJA dynamic_forall example on the host \n"); }
  if(exec_space.compare("host-parallel") == 0)
    { dynamic_policy = RAJA::expt::host_parallel; printf("Running RAJA dynamic_forall example on the host-parallel \n"); }
  if(exec_space.compare("device") == 0)
    { dynamic_policy = RAJA::expt::device; printf("Running RAJA dynamic_forall example on the device \n"); }

  //Users can provide 3 policies, host seq, host parallel, device
  using policy_list = camp::list<RAJA::loop_exec,
                                 RAJA::omp_parallel_for_exec,
                                 RAJA::cuda_exec<256>>;
  

  //policy is chosen from the list
  RAJA::expt::dynamic_forall<policy_list>(dynamic_policy, RAJA::RangeSegment(0, N), [=] RAJA_HOST_DEVICE (int i) {

      c[i] = a[i] + b[i];

  });


  // _rajaseq_vector_add_end

  checkResult(c, N);
//printResult(c, N);


//----------------------------------------------------------------------------//
//
// Clean up.
//
  memoryManager::deallocate(a);
  memoryManager::deallocate(b);
  memoryManager::deallocate(c);

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

