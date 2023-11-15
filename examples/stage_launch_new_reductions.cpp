//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <iostream>
#include <limits>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

/*
 *  Reduction Example
 *
 *  This example illustrates use of the RAJA reduction types: min, max,
 *  sum, min-loc, and max-loc.
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
  Specify the number of threads in a GPU thread block
*/
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP) || defined(RAJA_ENABLE_SYCL)
constexpr int DEVICE_BLOCK_SIZE = 256;
#endif

template<typename launch_pol, typename loop_pol>
 void test_new_launch_code(RAJA::TypedRangeSegment<int> arange, int *a, int N)
{

  int new_launch_sum = 0;
  const int no_teams = (N-1)/DEVICE_BLOCK_SIZE + 1;

  using reduce_policy = RAJA::cuda_reduce;
  RAJA::ReduceSum<reduce_policy, int> old_reducer_sum(0);


  RAJA::launch<launch_pol>
    (RAJA::LaunchParams(RAJA::Teams(no_teams),RAJA::Threads(DEVICE_BLOCK_SIZE)),
     "new_reduce_kernel",
     RAJA::expt::Reduce<RAJA::operators::plus>(&new_launch_sum),
     [=] RAJA_HOST_DEVICE (RAJA::LaunchContext ctx, int &_seq_sum)
     {

       RAJA::loop<loop_pol>(ctx, arange, [&] (int i) {
           _seq_sum += a[i];
           old_reducer_sum += a[i];
       });

        RAJA::loop<loop_pol>(ctx, arange, [&]  (int i) {
            _seq_sum += 1.0;
            old_reducer_sum += 1.0;
        });
     });


  std::cout << "test code 1: expected sum N = "<< N <<" | launch tsum = "
            << new_launch_sum << " old reducer sum = "<< old_reducer_sum << std::endl;

  RAJA::launch<launch_pol>
    (RAJA::LaunchParams(RAJA::Teams(no_teams),RAJA::Threads(DEVICE_BLOCK_SIZE)),
     "new_reduce_kernel",
     [=] RAJA_HOST_DEVICE (RAJA::LaunchContext ctx)
     {

       RAJA::loop<loop_pol>(ctx, arange, [&] (int i) {
           old_reducer_sum += a[i];
       });

        RAJA::loop<loop_pol>(ctx, arange, [&]  (int i) {
            old_reducer_sum += 1.0;
        });


     });


  std::cout << "test code 1: expected sum N = "<< N <<" | launch tsum = "
            << " old reducer sum = "<< old_reducer_sum << std::endl;
}

template<typename launch_pol, typename loop_pol>
void test_new_resource_launch_code(RAJA::resources::Resource &res, RAJA::TypedRangeSegment<int> arange, int *a, int N)
{

  int new_launch_sum = 0;
  const int no_teams = (N-1)/DEVICE_BLOCK_SIZE + 1;

  using reduce_policy = RAJA::cuda_reduce;
  RAJA::ReduceSum<reduce_policy, int> old_reducer_sum(0);


  RAJA::launch<launch_pol>
    (res, RAJA::LaunchParams(RAJA::Teams(no_teams),RAJA::Threads(DEVICE_BLOCK_SIZE)),
     "new_reduce_kernel",
     RAJA::expt::Reduce<RAJA::operators::plus>(&new_launch_sum),
     [=] RAJA_HOST_DEVICE (RAJA::LaunchContext ctx, int &_seq_sum)
     {

       RAJA::loop<loop_pol>(ctx, arange, [&] (int i) {
           _seq_sum += a[i];
           old_reducer_sum += a[i];
       });

        RAJA::loop<loop_pol>(ctx, arange, [&]  (int i) {
            _seq_sum += 1.0;
            old_reducer_sum += 1.0;
        });
     });


  std::cout << "resource test code 1: expected sum N = "<< N <<" | launch tsum = "
            << new_launch_sum << " old reducer sum = "<< old_reducer_sum << std::endl;

  RAJA::launch<launch_pol>
    (res, RAJA::LaunchParams(RAJA::Teams(no_teams),RAJA::Threads(DEVICE_BLOCK_SIZE)),
     "new_reduce_kernel",
     [=] RAJA_HOST_DEVICE (RAJA::LaunchContext ctx)
     {

       RAJA::loop<loop_pol>(ctx, arange, [&] (int i) {
           old_reducer_sum += a[i];
       });

        RAJA::loop<loop_pol>(ctx, arange, [&]  (int i) {
            old_reducer_sum += 1.0;
        });
     });

  std::cout << "resource test code 1: expected sum N = "<< N <<" | launch tsum = "
            << " old reducer sum = "<< old_reducer_sum << std::endl;
}


int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA reductions example...\n";

  // _reductions_array_init_start
//
// Define array length
//
  constexpr int N = 1000000;

//
// Allocate array data and initialize data to alternating sequence of 1, -1.
//
  int* a = memoryManager::allocate<int>(N);

  for (int i = 0; i < N; ++i) {
    if ( i % 2 == 0 ) {
      a[i] = 1;
    } else {
      a[i] = -1;
    }
  }

//
// Set min and max loc values
//
  constexpr int minloc_ref = N / 2;
  a[minloc_ref] = -100;

  constexpr int maxloc_ref = N / 2 + 1;
  a[maxloc_ref] = 100;
  // _reductions_array_init_end

//
// Note: with this data initialization scheme, the following results will
//       be observed for all reduction kernels below:
//
//  - the sum will be zero
//  - the min will be -100
//  - the max will be 100
//  - the min loc will be N/2
//  - the max loc will be N/2 + 1
//
//

//
// Define index range for iterating over a elements in all examples
//
  // _reductions_range_start
  RAJA::TypedRangeSegment<int> arange(0, N);
  // _reductions_range_end

//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA sequential reductions...\n";

  // _reductions_raja_seq_start
  using EXEC_POL1   = RAJA::seq_exec;

  int seq_sum = 0;

  RAJA::forall<EXEC_POL1>
    (arange, RAJA::expt::Reduce<RAJA::operators::plus>(&seq_sum), [=](int i, int &_seq_sum) {

      _seq_sum += a[i];

    });

  RAJA::forall<EXEC_POL1>
    (arange, RAJA::expt::Reduce<RAJA::operators::plus>(&seq_sum), [=](int i, int &_seq_sum) {

      _seq_sum += 1.0;

    });

  std::cout << "expected sum N = "<< N <<" | tsum = " << seq_sum << std::endl;
    // _reductions_raja_seq_end

//----------------------------------------------------------------------------//
  
  std::cout << "\n Seq | Running new RAJA reductions with launch...\n";

#if defined(RAJA_ENABLE_CUDA)
  {
    using gpu_launch_pol = RAJA::LaunchPolicy<RAJA::cuda_launch_t<false>>;
    using gpu_loop_pol = RAJA::LoopPolicy<RAJA::cuda_global_thread_x>;
    
    test_new_launch_code<gpu_launch_pol, gpu_loop_pol>(arange, a, N);
    
    RAJA::resources::Host host_res;
    RAJA::resources::Cuda device_res;
    RAJA::resources::Resource res = RAJA::Get_Runtime_Resource(host_res, device_res, RAJA::ExecPlace::DEVICE);
    
    test_new_resource_launch_code<gpu_launch_pol, gpu_loop_pol>(res, arange, a, N);
  }
  
#endif

//----------------------------------------------------------------------------//  

  

  


//
// Clean up.
//
  memoryManager::deallocate(a);

  std::cout << "\n DONE!...\n";

  return 0;
}
