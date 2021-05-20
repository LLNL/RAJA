//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <limits>
#include <vector>

#include "halo-exchange.hpp"
#include "Schedule.hpp"
#include "CopyTransaction.hpp"
#include "../memoryManager.hpp"
#include "loop.hpp"

#include "RAJA/util/Timer.hpp"


/*
 *  Halo exchange Example
 *
 *  Packs and Unpacks data from 3D variables as is done in a halo exchange.
 *  It illustrates how to use the workgroup set of constructs.
 *
 *  RAJA features shown:
 *    - `WorkPool` template object
 *    - `WorkGroup` template object
 *    - `WorkSite` template object
 *    -  Index range segment
 *    -  WorkGroup policies
 *
 * If CUDA is enabled, CUDA unified memory is used.
 */


//
// Functions for checking and printing results
//
void checkResult(std::vector<double*> const& vars, std::vector<double*> const& vars_ref,
                 int var_size, int num_vars);
void printResult(std::vector<double*> const& vars, int var_size, int num_vars);


int main(int argc, char **argv)
{

  std::cout << "\n\nRAJA halo exchange example...\n";

  if (argc != 1 && argc != 7) {
    std::cerr << "Usage: tut_halo-exchange "
              << "[grid_x grid_y grid_z halo_width num_vars num_cycles]\n";
    std::exit(1);
  }

  //
  // Define grid dimensions
  // Define halo width
  // Define number of grid variables
  // Define number of cycles
  //
  const int grid_dims[3] = { (argc != 7) ? 100 : std::atoi(argv[1]),
                             (argc != 7) ? 100 : std::atoi(argv[2]),
                             (argc != 7) ? 100 : std::atoi(argv[3]) };
  const int halo_width =     (argc != 7) ?   1 : std::atoi(argv[4]);
  const int num_vars   =     (argc != 7) ?   3 : std::atoi(argv[5]);
  const int num_cycles =     (argc != 7) ? 128 : std::atoi(argv[6]);

  std::cout << "grid dimensions "     << grid_dims[0]
            << " x "                  << grid_dims[1]
            << " x "                  << grid_dims[2] << "\n"
            << "halo width "          << halo_width   << "\n"
            << "number of variables " << num_vars     << "\n"
            << "number of cycles "    << num_cycles   << "\n";

  if ( grid_dims[0] < halo_width ||
       grid_dims[1] < halo_width ||
       grid_dims[2] < halo_width ) {
    std::cerr << "Error: "
              << "grid dimensions must not be smaller than the halo width\n";
    std::exit(1);
  }

  const int grid_plus_halo_dims[3] = { grid_dims[0] + 2*halo_width,
                                       grid_dims[1] + 2*halo_width,
                                       grid_dims[2] + 2*halo_width };

  const int var_size = grid_plus_halo_dims[0] *
                       grid_plus_halo_dims[1] *
                       grid_plus_halo_dims[2] ;

  //
  // Allocate grid variables and reference grid variables used to check
  // correctness.
  //
  std::vector<double*> vars    (num_vars, nullptr);
  std::vector<double*> vars_ref(num_vars, nullptr);

  for (int v = 0; v < num_vars; ++v) {
    vars[v]     = memoryManager::allocate<double>(var_size);
    vars_ref[v] = memoryManager::allocate<double>(var_size);
  }


  //
  // Generate index lists for packing and unpacking
  //
  std::vector<int*> pack_index_lists(num_neighbors, nullptr);
  std::vector<int > pack_index_list_lengths(num_neighbors, 0);
  create_pack_lists(pack_index_lists, pack_index_list_lengths, halo_width, grid_dims);

  std::vector<int*> unpack_index_lists(num_neighbors, nullptr);
  std::vector<int > unpack_index_list_lengths(num_neighbors, 0);
  create_unpack_lists(unpack_index_lists, unpack_index_list_lengths, halo_width, grid_dims);


  //
  // Convenience type alias to reduce typing
  //
  using range_segment = RAJA::TypedRangeSegment<int>;


  TimerStats timer;


//----------------------------------------------------------------------------//
  for (int p = static_cast<int>(LoopPattern::seq);
       p < static_cast<int>(LoopPattern::End); ++p) {

    LoopPattern pattern = static_cast<LoopPattern>(p);

    SetLoopPatternScope sepc(pattern);

    std::cout << "\n Running " << get_loop_pattern_name() << " halo exchange...\n";


    // allocate per pattern memory
    RAJA::resources::Resource res = get_loop_pattern_resource();

    std::vector<double*> pattern_vars(num_vars, nullptr);
    std::vector<int*>    pattern_pack_index_lists(num_neighbors, nullptr);
    std::vector<int*>    pattern_unpack_index_lists(num_neighbors, nullptr);

    for (int v = 0; v < num_vars; ++v) {
      pattern_vars[v] = res.allocate<double>(var_size);
    }

    for (int l = 0; l < num_neighbors; ++l) {
      int pack_len = pack_index_list_lengths[l];
      pattern_pack_index_lists[l] = res.allocate<int>(pack_len);
      res.memcpy(pattern_pack_index_lists[l], pack_index_lists[l], pack_len * sizeof(int));

      int unpack_len = unpack_index_list_lengths[l];
      pattern_unpack_index_lists[l] = res.allocate<int>(unpack_len);
      res.memcpy(pattern_unpack_index_lists[l], unpack_index_lists[l], unpack_len * sizeof(int));
    }


    const int my_rank = 0;
    Schedule schedule(my_rank);

    // populate schedule
    for (int l = 0; l < num_neighbors; ++l) {

      int neighbor_rank = l + 1;

      for (double* var : pattern_vars) {

        int* pack_list = pattern_pack_index_lists[l];
        int  pack_len  = pack_index_list_lengths[l];

        int* recv_list = pattern_unpack_index_lists[l];
        int  recv_len  = unpack_index_list_lengths[l];

        std::unique_ptr<Transaction> recv(
            new CopyTransaction(neighbor_rank,
                                my_rank,
                                var,
                                recv_list, recv_len));

        std::unique_ptr<Transaction> pack(
            new CopyTransaction(my_rank,
                                neighbor_rank,
                                var,
                                pack_list, pack_len));

        schedule.appendTransaction(std::move(recv));
        schedule.appendTransaction(std::move(pack));

      }

    }

    for (int c = 0; c < num_cycles; ++c ) {
      timer.start();
      {

        // set vars
        for (int v = 0; v < num_vars; ++v) {

          double* var = pattern_vars[v];

          loop(var_size, [=] RAJA_HOST_DEVICE (int i) {
            var[i] = i + v;
          });
        }

        schedule.communicate();

      }
      timer.stop();
    }

    // deallocate per pattern memory
    for (int v = 0; v < num_vars; ++v) {
      res.memcpy(vars[v], pattern_vars[v], var_size * sizeof(double));
      res.deallocate(pattern_vars[v]);
    }

    for (int l = 0; l < num_neighbors; ++l) {
      res.deallocate(pattern_pack_index_lists[l]);
      res.deallocate(pattern_unpack_index_lists[l]);
    }


    std::cout<< "\t" << timer.get_num() << " cycles" << std::endl;
    std::cout<< "\tavg cycle run time " << timer.get_avg() << " seconds" << std::endl;
    std::cout<< "\tmin cycle run time " << timer.get_min() << " seconds" << std::endl;
    std::cout<< "\tmax cycle run time " << timer.get_max() << " seconds" << std::endl;
    timer.reset();

    if (pattern == LoopPattern::seq) {
      // copy result of exchange for reference later
      for (int v = 0; v < num_vars; ++v) {
        res.memcpy(vars_ref[v], vars[v], var_size * sizeof(double));
      }
    } else {
      // check results against reference copy
      checkResult(vars, vars_ref, var_size, num_vars);
      //printResult(vars, var_size, num_vars);
    }
  }

#if 0

//----------------------------------------------------------------------------//
// RAJA::WorkGroup with allows deferred execution
// This has overhead and indirection not in the separate loop version,
// but can be useful for debugging.
//----------------------------------------------------------------------------//
  {
  std::cout << "\n Running RAJA loop workgroup halo exchange...\n";

    double minCycle = std::numeric_limits<double>::max();

    // _halo_exchange_loop_workgroup_policies_start
    using forall_policy = RAJA::loop_exec;

    using workgroup_policy = RAJA::WorkGroupPolicy <
                                 RAJA::loop_work,
                                 RAJA::ordered,
                                 RAJA::ragged_array_of_objects >;

    using workpool = RAJA::WorkPool< workgroup_policy,
                                     int,
                                     RAJA::xargs<>,
                                     memory_manager_allocator<char> >;

    using workgroup = RAJA::WorkGroup< workgroup_policy,
                                       int,
                                       RAJA::xargs<>,
                                       memory_manager_allocator<char> >;

    using worksite = RAJA::WorkSite< workgroup_policy,
                                     int,
                                     RAJA::xargs<>,
                                     memory_manager_allocator<char> >;
    // _halo_exchange_loop_workgroup_policies_end

    std::vector<double*> buffers(num_neighbors, nullptr);

    for (int l = 0; l < num_neighbors; ++l) {

      int buffer_len = num_vars * pack_index_list_lengths[l];

      buffers[l] = memoryManager::allocate<double>(buffer_len);

    }

    workpool pool_pack  (memory_manager_allocator<char>{});
    workpool pool_unpack(memory_manager_allocator<char>{});

    for (int c = 0; c < num_cycles; ++c ) {
      timer.start();
      {

      // set vars
      for (int v = 0; v < num_vars; ++v) {

        double* var = vars[v];

        RAJA::forall<forall_policy>(range_segment(0, var_size), [=] (int i) {
          var[i] = i + v;
        });
      }

      // _halo_exchange_loop_workgroup_packing_start
      for (int l = 0; l < num_neighbors; ++l) {

        double* buffer = buffers[l];
        int* list = pack_index_lists[l];
        int  len  = pack_index_list_lengths[l];

        // pack
        for (int v = 0; v < num_vars; ++v) {

          double* var = vars[v];

          pool_pack.enqueue(range_segment(0, len), [=] (int i) {
            buffer[i] = var[list[i]];
          });

          buffer += len;
        }
      }

      workgroup group_pack = pool_pack.instantiate();

      worksite site_pack = group_pack.run();

      // send all messages
      // _halo_exchange_loop_workgroup_packing_end

      // _halo_exchange_loop_workgroup_unpacking_start
      // recv all messages

      for (int l = 0; l < num_neighbors; ++l) {

        double* buffer = buffers[l];
        int* list = unpack_index_lists[l];
        int  len  = unpack_index_list_lengths[l];

        // unpack
        for (int v = 0; v < num_vars; ++v) {

          double* var = vars[v];

          pool_unpack.enqueue(range_segment(0, len), [=] (int i) {
            var[list[i]] = buffer[i];
          });

          buffer += len;
        }
      }

      workgroup group_unpack = pool_unpack.instantiate();

      worksite site_unpack = group_unpack.run();
      // _halo_exchange_loop_workgroup_unpacking_end

      }
      timer.stop();

      RAJA::Timer::ElapsedType tCycle = timer.elapsed();
      if (tCycle < minCycle) minCycle = tCycle;
      timer.reset();
    }

    for (int l = 0; l < num_neighbors; ++l) {

      memoryManager::deallocate(buffers[l]);

    }

    std::cout<< "\tmin cycle run time : " << minCycle << " seconds" << std::endl;

    // check results against reference copy
    checkResult(vars, vars_ref, var_size, num_vars);
    //printResult(vars, var_size, num_vars);
  }

//----------------------------------------------------------------------------//


#if defined(RAJA_ENABLE_OPENMP)


//----------------------------------------------------------------------------//
// RAJA::WorkGroup may allow effective parallelism across loops with Openmp.
//----------------------------------------------------------------------------//
  {
    std::cout << "\n Running RAJA OpenMP workgroup halo exchange...\n";

    double minCycle = std::numeric_limits<double>::max();

    // _halo_exchange_openmp_workgroup_policies_start
    using forall_policy = RAJA::omp_parallel_for_exec;

    using workgroup_policy = RAJA::WorkGroupPolicy <
                                 RAJA::omp_work,
                                 RAJA::ordered,
                                 RAJA::ragged_array_of_objects >;

    using workpool = RAJA::WorkPool< workgroup_policy,
                                     int,
                                     RAJA::xargs<>,
                                     memory_manager_allocator<char> >;

    using workgroup = RAJA::WorkGroup< workgroup_policy,
                                       int,
                                       RAJA::xargs<>,
                                       memory_manager_allocator<char> >;

    using worksite = RAJA::WorkSite< workgroup_policy,
                                     int,
                                     RAJA::xargs<>,
                                     memory_manager_allocator<char> >;
    // _halo_exchange_openmp_workgroup_policies_end

    std::vector<double*> buffers(num_neighbors, nullptr);

    for (int l = 0; l < num_neighbors; ++l) {

      int buffer_len = num_vars * pack_index_list_lengths[l];

      buffers[l] = memoryManager::allocate<double>(buffer_len);

    }

    workpool pool_pack  (memory_manager_allocator<char>{});
    workpool pool_unpack(memory_manager_allocator<char>{});

    for (int c = 0; c < num_cycles; ++c ) {
      timer.start();
      {

      // set vars
      for (int v = 0; v < num_vars; ++v) {

        double* var = vars[v];

        RAJA::forall<forall_policy>(range_segment(0, var_size), [=] (int i) {
          var[i] = i + v;
        });
      }

      // _halo_exchange_openmp_workgroup_packing_start
      for (int l = 0; l < num_neighbors; ++l) {

        double* buffer = buffers[l];
        int* list = pack_index_lists[l];
        int  len  = pack_index_list_lengths[l];

        // pack
        for (int v = 0; v < num_vars; ++v) {

          double* var = vars[v];

          pool_pack.enqueue(range_segment(0, len), [=] (int i) {
            buffer[i] = var[list[i]];
          });

          buffer += len;
        }
      }

      workgroup group_pack = pool_pack.instantiate();

      worksite site_pack = group_pack.run();

      // send all messages
      // _halo_exchange_openmp_workgroup_packing_end

      // _halo_exchange_openmp_workgroup_unpacking_start
      // recv all messages

      for (int l = 0; l < num_neighbors; ++l) {

        double* buffer = buffers[l];
        int* list = unpack_index_lists[l];
        int  len  = unpack_index_list_lengths[l];

        // unpack
        for (int v = 0; v < num_vars; ++v) {

          double* var = vars[v];

          pool_unpack.enqueue(range_segment(0, len), [=] (int i) {
            var[list[i]] = buffer[i];
          });

          buffer += len;
        }
      }

      workgroup group_unpack = pool_unpack.instantiate();

      worksite site_unpack = group_unpack.run();
      // _halo_exchange_openmp_workgroup_unpacking_end

      }
      timer.stop();

      RAJA::Timer::ElapsedType tCycle = timer.elapsed();
      if (tCycle < minCycle) minCycle = tCycle;
      timer.reset();
    }

    for (int l = 0; l < num_neighbors; ++l) {

      memoryManager::deallocate(buffers[l]);

    }

    std::cout<< "\tmin cycle run time : " << minCycle << " seconds" << std::endl;

    // check results against reference copy
    checkResult(vars, vars_ref, var_size, num_vars);
    //printResult(vars, var_size, num_vars);
  }

#endif

//----------------------------------------------------------------------------//


#if defined(RAJA_ENABLE_CUDA)

//----------------------------------------------------------------------------//
// RAJA::WorkGroup with cuda_work allows deferred kernel fusion execution
//----------------------------------------------------------------------------//
  {
    std::cout << "\n Running RAJA Cuda workgroup halo exchange...\n";

    double minCycle = std::numeric_limits<double>::max();


    std::vector<double*> cuda_vars(num_vars, nullptr);
    std::vector<int*>    cuda_pack_index_lists(num_neighbors, nullptr);
    std::vector<int*>    cuda_unpack_index_lists(num_neighbors, nullptr);

    for (int v = 0; v < num_vars; ++v) {
      cuda_vars[v] = memoryManager::allocate_gpu<double>(var_size);
    }

    for (int l = 0; l < num_neighbors; ++l) {
      int pack_len = pack_index_list_lengths[l];
      cuda_pack_index_lists[l] = memoryManager::allocate_gpu<int>(pack_len);
      cudaErrchk(cudaMemcpy( cuda_pack_index_lists[l], pack_index_lists[l], pack_len * sizeof(int), cudaMemcpyDefault ));

      int unpack_len = unpack_index_list_lengths[l];
      cuda_unpack_index_lists[l] = memoryManager::allocate_gpu<int>(unpack_len);
      cudaErrchk(cudaMemcpy( cuda_unpack_index_lists[l], unpack_index_lists[l], unpack_len * sizeof(int), cudaMemcpyDefault ));
    }

    std::swap(vars,               cuda_vars);
    std::swap(pack_index_lists,   cuda_pack_index_lists);
    std::swap(unpack_index_lists, cuda_unpack_index_lists);


    // _halo_exchange_cuda_workgroup_policies_start
    using forall_policy = RAJA::cuda_exec_async<CUDA_BLOCK_SIZE>;

    using workgroup_policy = RAJA::WorkGroupPolicy <
                                 RAJA::cuda_work_async<CUDA_WORKGROUP_BLOCK_SIZE>,
                                 RAJA::unordered_cuda_loop_y_block_iter_x_threadblock_average,
                                 RAJA::constant_stride_array_of_objects >;

    using workpool = RAJA::WorkPool< workgroup_policy,
                                     int,
                                     RAJA::xargs<>,
                                     pinned_allocator<char> >;

    using workgroup = RAJA::WorkGroup< workgroup_policy,
                                       int,
                                       RAJA::xargs<>,
                                       pinned_allocator<char> >;

    using worksite = RAJA::WorkSite< workgroup_policy,
                                     int,
                                     RAJA::xargs<>,
                                     pinned_allocator<char> >;
    // _halo_exchange_cuda_workgroup_policies_end

    std::vector<double*> buffers(num_neighbors, nullptr);

    for (int l = 0; l < num_neighbors; ++l) {

      int buffer_len = num_vars * pack_index_list_lengths[l];

      buffers[l] = memoryManager::allocate_gpu<double>(buffer_len);

    }

    workpool pool_pack  (pinned_allocator<char>{});
    workpool pool_unpack(pinned_allocator<char>{});

    for (int c = 0; c < num_cycles; ++c ) {
      timer.start();
      {

      // set vars
      for (int v = 0; v < num_vars; ++v) {

        double* var = vars[v];

        RAJA::forall<forall_policy>(range_segment(0, var_size), [=] RAJA_DEVICE (int i) {
          var[i] = i + v;
        });
      }

      // _halo_exchange_cuda_workgroup_packing_start
      for (int l = 0; l < num_neighbors; ++l) {

        double* buffer = buffers[l];
        int* list = pack_index_lists[l];
        int  len  = pack_index_list_lengths[l];

        // pack
        for (int v = 0; v < num_vars; ++v) {

          double* var = vars[v];

          pool_pack.enqueue(range_segment(0, len), [=] RAJA_DEVICE (int i) {
            buffer[i] = var[list[i]];
          });

          buffer += len;
        }
      }

      workgroup group_pack = pool_pack.instantiate();

      worksite site_pack = group_pack.run();

      cudaErrchk(cudaDeviceSynchronize());

      // send all messages
      // _halo_exchange_cuda_workgroup_packing_end

      // _halo_exchange_cuda_workgroup_unpacking_start
      // recv all messages

      for (int l = 0; l < num_neighbors; ++l) {

        double* buffer = buffers[l];
        int* list = unpack_index_lists[l];
        int  len  = unpack_index_list_lengths[l];

        // unpack
        for (int v = 0; v < num_vars; ++v) {

          double* var = vars[v];

          pool_unpack.enqueue(range_segment(0, len), [=] RAJA_DEVICE (int i) {
            var[list[i]] = buffer[i];
          });

          buffer += len;
        }
      }

      workgroup group_unpack = pool_unpack.instantiate();

      worksite site_unpack = group_unpack.run();

      cudaErrchk(cudaDeviceSynchronize());
      // _halo_exchange_cuda_workgroup_unpacking_end

      }
      timer.stop();

      RAJA::Timer::ElapsedType tCycle = timer.elapsed();
      if (tCycle < minCycle) minCycle = tCycle;
      timer.reset();
    }

    for (int l = 0; l < num_neighbors; ++l) {

      memoryManager::deallocate_gpu(buffers[l]);

    }


    std::swap(vars,               cuda_vars);
    std::swap(pack_index_lists,   cuda_pack_index_lists);
    std::swap(unpack_index_lists, cuda_unpack_index_lists);

    for (int v = 0; v < num_vars; ++v) {
      cudaErrchk(cudaMemcpy( vars[v], cuda_vars[v], var_size * sizeof(double), cudaMemcpyDefault ));
      memoryManager::deallocate_gpu(cuda_vars[v]);
    }

    for (int l = 0; l < num_neighbors; ++l) {
      memoryManager::deallocate_gpu(cuda_pack_index_lists[l]);
      memoryManager::deallocate_gpu(cuda_unpack_index_lists[l]);
    }


    std::cout<< "\tmin cycle run time : " << minCycle << " seconds" << std::endl;

    // check results against reference copy
    checkResult(vars, vars_ref, var_size, num_vars);
    //printResult(vars, var_size, num_vars);
  }

#endif

//----------------------------------------------------------------------------//


#if defined(RAJA_ENABLE_HIP)

//----------------------------------------------------------------------------//
// RAJA::WorkGroup with hip_work allows deferred kernel fusion execution
//----------------------------------------------------------------------------//
  {
    std::cout << "\n Running RAJA Hip workgroup halo exchange...\n";

    double minCycle = std::numeric_limits<double>::max();


    std::vector<double*> hip_vars(num_vars, nullptr);
    std::vector<int*>    hip_pack_index_lists(num_neighbors, nullptr);
    std::vector<int*>    hip_unpack_index_lists(num_neighbors, nullptr);

    for (int v = 0; v < num_vars; ++v) {
      hip_vars[v] = memoryManager::allocate_gpu<double>(var_size);
    }

    for (int l = 0; l < num_neighbors; ++l) {
      int pack_len = pack_index_list_lengths[l];
      hip_pack_index_lists[l] = memoryManager::allocate_gpu<int>(pack_len);
      hipErrchk(hipMemcpy( hip_pack_index_lists[l], pack_index_lists[l], pack_len * sizeof(int), hipMemcpyHostToDevice ));

      int unpack_len = unpack_index_list_lengths[l];
      hip_unpack_index_lists[l] = memoryManager::allocate_gpu<int>(unpack_len);
      hipErrchk(hipMemcpy( hip_unpack_index_lists[l], unpack_index_lists[l], unpack_len * sizeof(int), hipMemcpyHostToDevice ));
    }

    std::swap(vars,               hip_vars);
    std::swap(pack_index_lists,   hip_pack_index_lists);
    std::swap(unpack_index_lists, hip_unpack_index_lists);


    // _halo_exchange_hip_workgroup_policies_start
    using forall_policy = RAJA::hip_exec_async<HIP_BLOCK_SIZE>;

    using workgroup_policy = RAJA::WorkGroupPolicy <
                                 RAJA::hip_work_async<HIP_WORKGROUP_BLOCK_SIZE>,
#if defined(RAJA_ENABLE_HIP_INDIRECT_FUNCTION_CALL)
                                 RAJA::unordered_hip_loop_y_block_iter_x_threadblock_average,
#else
                                 RAJA::ordered,
#endif
                                 RAJA::constant_stride_array_of_objects >;

    using workpool = RAJA::WorkPool< workgroup_policy,
                                     int,
                                     RAJA::xargs<>,
                                     pinned_allocator<char> >;

    using workgroup = RAJA::WorkGroup< workgroup_policy,
                                       int,
                                       RAJA::xargs<>,
                                       pinned_allocator<char> >;

    using worksite = RAJA::WorkSite< workgroup_policy,
                                     int,
                                     RAJA::xargs<>,
                                     pinned_allocator<char> >;
    // _halo_exchange_hip_workgroup_policies_end

    std::vector<double*> buffers(num_neighbors, nullptr);

    for (int l = 0; l < num_neighbors; ++l) {

      int buffer_len = num_vars * pack_index_list_lengths[l];

      buffers[l] = memoryManager::allocate_gpu<double>(buffer_len);

    }

    workpool pool_pack  (pinned_allocator<char>{});
    workpool pool_unpack(pinned_allocator<char>{});

    for (int c = 0; c < num_cycles; ++c ) {
      timer.start();
      {

      // set vars
      for (int v = 0; v < num_vars; ++v) {

        double* var = vars[v];

        RAJA::forall<forall_policy>(range_segment(0, var_size), [=] RAJA_DEVICE (int i) {
          var[i] = i + v;
        });
      }

      // _halo_exchange_hip_workgroup_packing_start
      for (int l = 0; l < num_neighbors; ++l) {

        double* buffer = buffers[l];
        int* list = pack_index_lists[l];
        int  len  = pack_index_list_lengths[l];

        // pack
        for (int v = 0; v < num_vars; ++v) {

          double* var = vars[v];

          pool_pack.enqueue(range_segment(0, len), [=] RAJA_DEVICE (int i) {
            buffer[i] = var[list[i]];
          });

          buffer += len;
        }
      }

      workgroup group_pack = pool_pack.instantiate();

      worksite site_pack = group_pack.run();

      hipErrchk(hipDeviceSynchronize());

      // send all messages
      // _halo_exchange_hip_workgroup_packing_end

      // _halo_exchange_hip_workgroup_unpacking_start
      // recv all messages

      for (int l = 0; l < num_neighbors; ++l) {

        double* buffer = buffers[l];
        int* list = unpack_index_lists[l];
        int  len  = unpack_index_list_lengths[l];

        // unpack
        for (int v = 0; v < num_vars; ++v) {

          double* var = vars[v];

          pool_unpack.enqueue(range_segment(0, len), [=] RAJA_DEVICE (int i) {
            var[list[i]] = buffer[i];
          });

          buffer += len;
        }
      }

      workgroup group_unpack = pool_unpack.instantiate();

      worksite site_unpack = group_unpack.run();

      hipErrchk(hipDeviceSynchronize());
      // _halo_exchange_hip_workgroup_unpacking_end

      }
      timer.stop();

      RAJA::Timer::ElapsedType tCycle = timer.elapsed();
      if (tCycle < minCycle) minCycle = tCycle;
      timer.reset();
    }

    for (int l = 0; l < num_neighbors; ++l) {

      memoryManager::deallocate_gpu(buffers[l]);

    }


    std::swap(vars,               hip_vars);
    std::swap(pack_index_lists,   hip_pack_index_lists);
    std::swap(unpack_index_lists, hip_unpack_index_lists);

    for (int v = 0; v < num_vars; ++v) {
      hipErrchk(hipMemcpy( vars[v], hip_vars[v], var_size * sizeof(double), hipMemcpyDeviceToHost ));
      memoryManager::deallocate_gpu(hip_vars[v]);
    }

    for (int l = 0; l < num_neighbors; ++l) {
      memoryManager::deallocate_gpu(hip_pack_index_lists[l]);
      memoryManager::deallocate_gpu(hip_unpack_index_lists[l]);
    }


    std::cout<< "\tmin cycle run time : " << minCycle << " seconds" << std::endl;

    // check results against reference copy
    checkResult(vars, vars_ref, var_size, num_vars);
    //printResult(vars, var_size, num_vars);
  }

#endif

//----------------------------------------------------------------------------//

#endif

//
// Clean up.
//
  for (int v = 0; v < num_vars; ++v) {
    memoryManager::deallocate(vars[v]);
    memoryManager::deallocate(vars_ref[v]);
  }

  destroy_pack_lists(pack_index_lists);
  destroy_unpack_lists(unpack_index_lists);


  std::cout << "\n DONE!...\n";

  return 0;
}


//
// Function to compare result to reference and report P/F.
//
void checkResult(std::vector<double*> const& vars, std::vector<double*> const& vars_ref,
                 int var_size, int num_vars)
{
  bool correct = true;
  for (int v = 0; v < num_vars; ++v) {
    double* var = vars[v];
    double* var_ref = vars_ref[v];
    for (int i = 0; i < var_size; i++) {
      if ( var[i] != var_ref[i] ) { correct = false; }
    }
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
void printResult(std::vector<double*> const& vars, int var_size, int num_vars)
{
  std::cout << std::endl;
  for (int v = 0; v < num_vars; ++v) {
    double* var = vars[v];
    for (int i = 0; i < var_size; i++) {
      std::cout << "result[" << i << "] = " << var[i] << std::endl;
    }
  }
  std::cout << std::endl;
}

