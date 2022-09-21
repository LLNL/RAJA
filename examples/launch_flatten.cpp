//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
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
 *  Thread Flatten Example using RAJA Launch
 *
 *  This example illustrates use of the "flatten"
 *  policy inside RAJA Launch
 *
 *  The flatten policy enables reshaping
 *  multi-dimensional thread teams to 1D
 *  simplifying indexing patterns
 *
 *  RAJA features shown:
 *    - `launch' method to create kernel
 *    -  flatten_block_threads
 *
 */

/*
 * Define device launch policies
 */

#if defined(RAJA_ENABLE_CUDA)
using device_launch = RAJA::expt::LaunchPolicy<RAJA::expt::cuda_launch_t<false>>;
using device_inner_pol0 =  RAJA::expt::LoopPolicy<RAJA::cuda_thread_x_direct>;
using device_inner_pol1 =  RAJA::expt::LoopPolicy<RAJA::cuda_thread_y_direct>;
using device_flatten_pol =  RAJA::expt::LoopPolicy<RAJA::expt::cuda_flatten_block_threads_xy_direct>;
using reduce_policy = RAJA::cuda_reduce;
#elif defined(RAJA_ENABLE_HIP)
using device_launch = RAJA::expt::LaunchPolicy<RAJA::expt::hip_launch_t<false>>;
using device_inner_pol0 =  RAJA::expt::LoopPolicy<RAJA::hip_thread_x_direct>;
using device_inner_pol1 =  RAJA::expt::LoopPolicy<RAJA::hip_thread_y_direct>;
using device_flatten_pol =  RAJA::expt::LoopPolicy<RAJA::expt::hip_flatten_block_threads_xy_direct>;
using reduce_policy = RAJA::hip_reduce;
#endif

/*
 * Define device launch policies
 */

using host_launch = RAJA::expt::LaunchPolicy<RAJA::expt::seq_launch_t>;
using host_loop = RAJA::expt::LoopPolicy<RAJA::loop_exec>;

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)

  //
  // Problem size dimensions
  //
  constexpr int N = 4;
  constexpr int NN = N*N;

  //
  // dynamic shared memory used in kernel
  //
  constexpr size_t dynamic_shared_mem = 0;

  //
  // Configure grid size
  //
  RAJA::expt::Grid grid(RAJA::expt::Teams(1),
                        RAJA::expt::Threads(N, N),
                        "Launch Flatten Kernel");

  //
  // Resource object for host, used to allocate memory
  //
  camp::resources::Host host_res;
  int *h_A_ptr = host_res.allocate<int>(NN);

  //
  // Resource object for device, used to allocate memory
  //
#if defined(RAJA_ENABLE_CUDA)
  camp::resources::Cuda device_res;
#elif defined(RAJA_ENABLE_HIP)
  camp::resources::Hip device_res;
#endif

  int *d_A_ptr = device_res.allocate<int>(NN);

//----------------------------------------------------------------------------//

  std::cout << "\n Running device version of teams_flatten example ...\n";

  RAJA::ReduceSum<reduce_policy, int> device_kernel_sum(0);
  RAJA::View<int, RAJA::Layout<2>> d_A_2DView(d_A_ptr, N, N);
  RAJA::View<int, RAJA::Layout<1>> d_A_1DView(d_A_ptr, NN);

  RAJA::expt::launch<device_launch>
    (dynamic_shared_mem, grid, [=] RAJA_HOST_DEVICE (RAJA::expt::LaunchContext ctx)
     {

       RAJA::expt::loop<device_inner_pol1>(ctx, RAJA::RangeSegment(0, N), [&] (int j) {
         RAJA::expt::loop<device_inner_pol0>(ctx, RAJA::RangeSegment(0, N), [&] (int i) {
             d_A_2DView(j, i) = i + j;
           });
         });

       ctx.teamSync();

       // RAJA flatten policy will reshape a 2/3D thread team to 1D simplifying
       // accumulating memory contents
       RAJA::expt::loop<device_flatten_pol>(ctx, RAJA::RangeSegment(0, NN), [&] (int i) {
           device_kernel_sum += d_A_1DView(i);
       });

     });

//----------------------------------------------------------------------------//

  std::cout << "\n Running host version of teams_flatten example ...\n";

  RAJA::ReduceSum<reduce_policy, int> host_kernel_sum(0);
  RAJA::View<int, RAJA::Layout<2>> h_A_2DView(h_A_ptr, N, N);
  RAJA::View<int, RAJA::Layout<1>> h_A_1DView(h_A_ptr, NN);

  RAJA::expt::launch<host_launch>
    (dynamic_shared_mem, grid, [=] RAJA_HOST_DEVICE (RAJA::expt::LaunchContext ctx)
    {

       RAJA::expt::loop<host_loop>(ctx, RAJA::RangeSegment(0, N), [&] (int j) {
         RAJA::expt::loop<host_loop>(ctx, RAJA::RangeSegment(0, N), [&] (int i) {
             h_A_2DView(j, i) = i + j;
           });
         });

       ctx.teamSync();

       //As loops are dispatched as standard C loops we can revert to using
       //a regular loop_exec policy
       RAJA::expt::loop<host_loop>(ctx, RAJA::RangeSegment(0, NN), [&] (int i) {
           host_kernel_sum += h_A_1DView(i);
       });

     });

  if ( device_kernel_sum.get() == host_kernel_sum.get() ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }

#else
  std::cout << "Please build with CUDA or Hip to run this example ...\n";
#endif

  std::cout << "\n DONE!...\n";

  return 0;
}
