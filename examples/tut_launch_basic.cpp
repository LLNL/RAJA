//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "RAJA/RAJA.hpp"

/*
 *  Developing with RAJA Launch
 *
 *  This example serves as a basic overview of
 *  capabilities with the RAJA Launch API.
 *
 *  RAJA features shown:
 *    -  RAJA::expt::launch
 *    -  RAJA::expt::loop
 */

/*
 * The RAJA::expt::Launch framework enables developers
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
// __device_launch_start
using device_launch = RAJA::expt::cuda_launch_t<false>;
// __device_launch_end
#elif defined(RAJA_ENABLE_HIP)
using device_launch = RAJA::expt::hip_launch_t<false>;
#endif

using launch_policy = RAJA::expt::LaunchPolicy<
  host_launch
#if defined(RAJA_DEVICE_ACTIVE)
  ,device_launch
#endif
  >;

/*
 * RAJA launch exposes a thread/block programming model
 * as used in CUDA/HIP/SYCL. Loops within an execution
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

#if defined(RAJA_DEVICE_ACTIVE)
__global__ void gpuKernel()
{
  //Equivalent CUDA/HIP style thread/block mapping
  // _device_loop_start
  {int by = blockIdx.y;
    {int bx = blockIdx.x;

      {int ty = threadIdx.y;
        {int tx = blockIdx.x;

          printf("device-iter: threadIdx_tx %d threadIdx_ty %d block_bx %d block_by %d \n",
                 tx, ty, bx, by);

        }
      }

    }
  }
  // _device_loop_end
}
#endif

int main(int argc, char *argv[])
{

  if(argc != 2) {
    RAJA_ABORT_OR_THROW("Usage ./tut_launch_basic host or ./tut_launch_basic device");
  }

//
// Run time policy section is demonstrated in this example by specifying
// kernel exection space as a command line argument (host or device).
// Example usage ./tut_launch_basic host or ./tut_launch_basic device
//
  std::string exec_space = argv[1];
  if(!(exec_space.compare("host") == 0 || exec_space.compare("device") == 0 )){
    RAJA_ABORT_OR_THROW("Usage ./tut_launch_basic host or ./tut_launch_basic device");
    return 0;
  }

  RAJA::expt::ExecPlace select_cpu_or_gpu;
  if(exec_space.compare("host") == 0)
    { select_cpu_or_gpu = RAJA::expt::HOST; printf("Running RAJA-Teams on the host \n"); }
  if(exec_space.compare("device") == 0)
    { select_cpu_or_gpu = RAJA::expt::DEVICE; printf("Running RAJA-Teams on the device \n"); }

//
// The following three kernels illustrate loop based parallelism
// based on nested for loops. For correctness team and thread loops
// make the assumption that all work inside can be done
// concurrently.
//

  // __compute_grid_start
  const int Nteams  = 2;
  const int Nthreads = 2;
  // __compute_grid_end
 
  RAJA::expt::launch<launch_policy>(select_cpu_or_gpu,
    RAJA::expt::Grid(RAJA::expt::Teams(Nteams,Nteams),
                     RAJA::expt::Threads(Nthreads,Nthreads)),

    [=] RAJA_HOST_DEVICE (RAJA::expt::LaunchContext ctx) {

    // _team_loops_start
     RAJA::expt::loop<teams_y>(ctx, RAJA::TypedRangeSegment<int>(0, Nteams), [&] (int by) {
       RAJA::expt::loop<teams_x>(ctx, RAJA::TypedRangeSegment<int>(0, Nteams), [&] (int bx) {

         RAJA::expt::loop<threads_y>(ctx, RAJA::TypedRangeSegment<int>(0, Nthreads), [&] (int ty) {
           RAJA::expt::loop<threads_x>(ctx, RAJA::TypedRangeSegment<int>(0, Nthreads),       [&] (int tx) {
               printf("RAJA Teams: threadId_x %d threadId_y %d teamId_x %d teamId_y %d \n",
                      tx, ty, bx, by);

           });
         });

       });
     });
     // _team_loops_end

   });

  //Equivalent C style loops
  if(select_cpu_or_gpu == RAJA::expt::HOST) {
    // _c_style_loops_start
    for(int by=0; by<Nteams; ++by) {
      for(int bx=0; bx<Nteams; ++bx) {

        for(int ty=0; ty<Nthreads; ++ty) {
          for(int tx=0; tx<Nthreads; ++tx) {

            printf("c-iter: iter_tx %d iter_ty %d iter_bx %d iter_by %d \n",
                   tx, ty, bx, by);
          }
        }

      }
    }
    // _c_style_loops_end
  }


//
// The following launches equivalent
// device kernels
//
#if defined(RAJA_DEVICE_ACTIVE)
  // Define thread block dimensions
  dim3 blockdim(Nthreads, Nthreads);
  // Define grid dimensions to match the RAJA version above
  dim3 griddim(Nteams, Nteams);
#endif

#if defined(RAJA_ENABLE_CUDA)
  if(select_cpu_or_gpu == RAJA::expt::DEVICE)
    gpuKernel<<<griddim, blockdim>>>();
  cudaDeviceSynchronize();
#endif

#if defined(RAJA_ENABLE_HIP)
  if(select_cpu_or_gpu == RAJA::expt::DEVICE)
    hipLaunchKernelGGL((gpuKernel), dim3(griddim), dim3(blockdim), 0, 0);
  hipDeviceSynchronize();
#endif

  return 0;
}
