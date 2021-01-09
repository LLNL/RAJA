//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "RAJA/RAJA.hpp"

/*
 *  Developing with RAJA Teams
 *
 *  This example serves as a basic overview of
 *  capabilities with the RAJA Teams API.
 *
 *  RAJA features shown:
 *    -  RAJA::expt::launch
 */

/*
 * The RAJA teams framework enables developers
 * to expressed algorithms in terms of nested
 * loops within an execution space. RAJA teams
 * enables run time selection of a host or
 * device execution space. As a starting point
 * the example below choses a sequential
 * execution space and either a CUDA or HIP
 * execution device execution space.
*/

// __host_launch_start
using host_launch = RAJA::expt::seq_launch_t;
// __host_launch_end

#if defined(RAJA_ENABLE_CUDA)
// __host_device_start
using device_launch = RAJA::expt::cuda_launch_t<false>;
// __host_device_end
#elif defined(RAJA_ENABLE_HIP)
using device_launch = RAJA::expt::cuda_launch_t<false>;
#endif

using launch_policy = RAJA::expt::LaunchPolicy<
  host_launch
#if defined(RAJA_DEVICE_ACTIVE)
  ,device_launch
#endif
  >;

/*
 * RAJA teams follows a similar thread/block programming model
 * as found in CUDA/HIP/SYCL. Loops within an execution
 * maybe mapped to either threads or teams. Under this
 * programming model, computation is performed with
 * a collection of threads which are grouped into teams.
 * This threading hierarchy enables us to express hierarchical
 * parallism. In the example below we define polices for 2D thread
 * teams (up to 3D) is supported, and a 2D grid of teams.
 * On the host the loops expands to standard C style for loops.
 */

using teams_x = RAJA::expt::LoopPolicy<
                                       RAJA::loop_exec
#if defined(RAJA_ENABLE_CUDA)
                                       ,
                                       RAJA::cuda_block_x_direct
#endif
#if defined(RAJA_ENABLE_HIP)
                                       ,
                                       RAJA::hip_block_x_direct
#endif
                                       >;

using teams_y = RAJA::expt::LoopPolicy<
                                       RAJA::loop_exec
#if defined(RAJA_ENABLE_CUDA)
                                       ,
                                       RAJA::cuda_block_y_direct
#endif
#if defined(RAJA_ENABLE_HIP)
                                       ,
                                       RAJA::hip_block_y_direct
#endif
                                       >;

using threads_x = RAJA::expt::LoopPolicy<RAJA::loop_exec
#if defined(RAJA_ENABLE_CUDA)
                                         ,
                                         RAJA::cuda_thread_x_direct
#endif
#if defined(RAJA_ENABLE_HIP)
                                         ,
                                         RAJA::hip_thread_x_direct
#endif
                                         >;

using threads_y = RAJA::expt::LoopPolicy<RAJA::loop_exec
#if defined(RAJA_ENABLE_CUDA)
                                         ,
                                         RAJA::cuda_thread_y_direct
#endif
#if defined(RAJA_ENABLE_HIP)
                                         ,
                                         RAJA::hip_thread_y_direct
#endif
                                         >;


int main(int argc, char *argv[])
{

  std::string exec_space = argv[1];
  if(!(exec_space.compare("host") == 0 || exec_space.compare("device") == 0 )){
    RAJA_ABORT_OR_THROW("Usage ./tut_teams_basic host or ./tut_teams_basic device");
    return 0;
  }

  RAJA::expt::ExecPlace select_cpu_or_gpu;
  if(exec_space.compare("host") == 0)
    { select_cpu_or_gpu = RAJA::expt::HOST; printf("Running RAJA-Teams on the host \n"); }
  if(exec_space.compare("device") == 0)
    { select_cpu_or_gpu = RAJA::expt::DEVICE; printf("Running RAJA-Teams on the device \n"); }

  const int Nteams  = 2;
  const int Nthreads = 2;

  RAJA::expt::launch<launch_policy>(select_cpu_or_gpu,
  RAJA::expt::Resources(RAJA::expt::Teams(Nteams,Nteams),
                        RAJA::expt::Threads(Nthreads,Nthreads)),
  [=] RAJA_HOST_DEVICE (RAJA::expt::LaunchContext ctx) {

  RAJA::expt::loop<teams_y>(ctx, RAJA::RangeSegment(0, Nteams), [&] (int by) {
    RAJA::expt::loop<teams_x>(ctx, RAJA::RangeSegment(0, Nteams), [&] (int bx) {

      RAJA::expt::loop<threads_y>(ctx, RAJA::RangeSegment(0, Nthreads), [&] (int ty) {
        RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(0, Nthreads), [&] (int tx) {

          printf("threadId_x %d threadId_y %d teamId_x %d teamId_y %d \n", tx, ty, bx, by);

          });
        });

      });
    });

 });

  return 0;
}
