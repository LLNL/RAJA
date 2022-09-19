//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "RAJA/RAJA.hpp"
#include "camp/resource.hpp"
#include "memoryManager.hpp"


/*
 * RAJA Launch Example: Upper Triangular Pattern + Shared Memory
 *
 * Launch introduces hierarchical parallelism through the concept of
 * teams and threads.  Computation is executed in a pre-defined grid
 * composed of threads and grouped into teams. The teams model enables
 * developers to express parallelism through loops over teams, and inner loops
 * over threads. Team loops are executed in parallel and
 * threads within a team should be treated as sub-parallel regions.
 *
 * Team shared memory is allocated between team and thread loops.
 * Memory allocated within thread loops are thread private.
 * The example below demonstrates composing an upper triangular
 * loop pattern, and using shared memory.
 *
 */

/*
 * Define host/device launch policies
 */
using launch_policy = RAJA::expt::LaunchPolicy<
#if defined(RAJA_ENABLE_OPENMP)
    RAJA::expt::omp_launch_t
#else
    RAJA::expt::seq_launch_t
#endif
#if defined(RAJA_ENABLE_CUDA)
    ,
    RAJA::expt::cuda_launch_t<false>
#endif
#if defined(RAJA_ENABLE_HIP)
    ,
    RAJA::expt::hip_launch_t<false>
#endif
#if defined(RAJA_ENABLE_SYCL)
    ,
    RAJA::expt::sycl_launch_t<false>
#endif
    >;

/*
 * Define team policies.
 * Up to 3 dimension are supported: x,y,z
 */
using teams_x = RAJA::expt::LoopPolicy<
#if defined(RAJA_ENABLE_OPENMP)
                                       RAJA::omp_parallel_for_exec
#else
                                       RAJA::loop_exec
#endif
#if defined(RAJA_ENABLE_CUDA)
                                       ,
                                       RAJA::cuda_block_x_direct
#endif
#if defined(RAJA_ENABLE_HIP)
                                       ,
                                       RAJA::hip_block_x_direct
#endif
#if defined(RAJA_ENABLE_SCYL)
                                       ,
                                       RAJA::sycl_group_0_direct
#endif
                                       >;
/*
 * Define thread policies.
 * Up to 3 dimension are supported: x,y,z
 */
using threads_x = RAJA::expt::LoopPolicy<RAJA::loop_exec
#if defined(RAJA_ENABLE_CUDA)
                                         ,
                                         RAJA::cuda_thread_x_loop
#endif
#if defined(RAJA_ENABLE_HIP)
                                         ,
                                         RAJA::hip_thread_x_loop
#endif
#if defined(RAJA_ENABLE_SYCL)
                                         ,
                                         RAJA::sycl_local_0_loop
#endif
                                         >;


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  // Resource object for host
  camp::resources::Host host_res;

  // Resource objects for CUDA or HIP
#if defined(RAJA_ENABLE_CUDA)
  camp::resources::Cuda device_res;
#endif

#if defined(RAJA_ENABLE_HIP)
  camp::resources::Hip device_res;
#endif

#if defined(RAJA_ENABLE_SYCL)
    memoryManager::sycl_res = new camp::resources::Resource{camp::resources::Sycl()};
  ::RAJA::sycl::detail::setQueue(memoryManager::sycl_res);
  camp::resources::Sycl device_res;
#endif

  std::cout << "\n Running RAJA-Launch examples...\n";
  int num_of_backends = 1;
#if defined(RAJA_DEVICE_ACTIVE)
  num_of_backends++;
#endif

  // RAJA teams may switch between host and device policies at run time.
  // The loop below will execute through the available backends.

  for (int exec_place = 0; exec_place < num_of_backends; ++exec_place) {

    RAJA::expt::ExecPlace select_cpu_or_gpu = (RAJA::expt::ExecPlace)exec_place;

    // auto select_cpu_or_gpu = RAJA::expt::HOST;
    // auto select_cpu_or_gpu = RAJA::expt::DEVICE;

    // Allocate memory for either host or device
    int N_tri = 5;

    int* Ddat = nullptr;
    if (select_cpu_or_gpu == RAJA::expt::HOST) {
      Ddat = host_res.allocate<int>(N_tri * N_tri);
    }

#if defined(RAJA_DEVICE_ACTIVE)
    if (select_cpu_or_gpu == RAJA::expt::DEVICE) {
      Ddat = device_res.allocate<int>(N_tri * N_tri);
    }
#endif

    /*
     * RAJA::expt::launch just starts a "kernel" and doesn't provide any looping.
     *
     * The first argument determines which policy should be executed,
     *
     * The second argument is the number of teams+threads needed for each of the
     * policies.
     *
     * Third argument is the lambda.
     *
     * The lambda takes a "resource" object, which has the teams+threads
     * and is used to perform thread synchronizations within a team.
     */

    if (select_cpu_or_gpu == RAJA::expt::HOST){
      std::cout << "\n Running upper triangular pattern example on the host...\n";
    }else {
      std::cout << "\n Running upper triangular pattern example on the device...\n";
    }


    RAJA::View<int, RAJA::Layout<2>> D(Ddat, N_tri, N_tri);

    const size_t shared_memory = sizeof(int);

    //TODO need to fix, segfaults if we cycle between cpu and gpu options

    RAJA::expt::launch<launch_policy>
      //(select_cpu_or_gpu,
       (RAJA::expt::DEVICE,
       RAJA::expt::Grid(RAJA::expt::Teams(N_tri),
			RAJA::expt::Threads(N_tri),
			shared_memory),
       [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

         RAJA::expt::loop<teams_x>(ctx, RAJA::RangeSegment(0, N_tri), [&](int r) {

	   // Array shared within threads of the same team
	   int *s_A = ctx.getSharedMemory<int>(1);

           RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(0, 1), [&](int c) {
              s_A[c] = r;
           });  // loop c

           ctx.teamSync();

           RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(r, N_tri), [&](int c) {
               D(r, c) = r * N_tri + c;

	       //SYCL does not support printf
               //printf("r=%d, c=%d : D=%d : s_A = %d \n", r, c, D(r, c), s_A[0]);
           });  // loop c

         });  // loop r

       });  // outer lambda

    if (select_cpu_or_gpu == RAJA::expt::HOST) {
      host_res.deallocate(Ddat);
    }

#if defined(RAJA_DEVICE_ACTIVE)
    if (select_cpu_or_gpu == RAJA::expt::DEVICE) {
      device_res.deallocate(Ddat);
    }
#endif

  }  // Execution places loop


}  // Main
