/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run kernel
 *          traversals on GPU with CUDA.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_cuda_kernel_CudaKernel_HPP
#define RAJA_policy_cuda_kernel_CudaKernel_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include <cassert>
#include <climits>

#include "camp/camp.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/pattern/kernel.hpp"
#include "RAJA/pattern/kernel/For.hpp"
#include "RAJA/pattern/kernel/Lambda.hpp"

#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#include "RAJA/policy/cuda/policy.hpp"

#include "RAJA/internal/LegacyCompatibility.hpp"

#include "RAJA/policy/cuda/kernel/internal.hpp"

namespace RAJA
{

/*!
 * CUDA kernel launch policy where the user may specify the number of physical
 * thread blocks and threads per block.
 * If num_blocks is 0 and num_threads is non-zero then num_blocks is chosen at
 * runtime.
 * Num_blocks is chosen to maximize the number of blocks running concurrently.
 * If num_threads and num_blocks are both 0 then num_threads and num_blocks are
 * chosen at runtime.
 * Num_threads and num_blocks are determined by the CUDA occupancy calculator.
 * If num_threads is 0 and num_blocks is non-zero then num_threads is chosen at
 * runtime.
 * Num_threads is 1024, which may not be appropriate for all kernels.
 */
template <bool async0, int num_blocks, int num_threads>
struct cuda_launch {};

/*!
 * CUDA kernel launch policy where the user specifies the number of physical
 * thread blocks and threads per block.
 * If num_blocks is 0 then num_blocks is chosen at runtime.
 * Num_blocks is chosen to maximize the number of blocks running concurrently.
 */
template <bool async0, int num_blocks, int num_threads>
using cuda_explicit_launch = cuda_launch<async0, num_blocks, num_threads>;


/*!
 * CUDA kernel launch policy where the number of physical blocks and threads
 * are determined by the CUDA occupancy calculator.
 * If num_threads is 0 then num_threads is chosen at runtime.
 */
template <int num_threads0, bool async0>
using cuda_occ_calc_launch = cuda_launch<async0, 0, num_threads0>;

namespace statement
{

/*!
 * A RAJA::kernel statement that launches a CUDA kernel.
 *
 *
 */
template <typename LaunchConfig, typename... EnclosedStmts>
struct CudaKernelExt
    : public internal::Statement<cuda_exec<0>, EnclosedStmts...> {
};


/*!
 * A RAJA::kernel statement that launches a CUDA kernel with the flexibility
 * to fix the number of threads and/or blocks and let the CUDA occupancy
 * calculator determine the unspecified values.
 * The kernel launch is synchronous.
 */
template <int num_blocks, int num_threads, typename... EnclosedStmts>
using CudaKernelExp =
    CudaKernelExt<cuda_launch<false, num_blocks, num_threads>, EnclosedStmts...>;

/*!
 * A RAJA::kernel statement that launches a CUDA kernel with the flexibility
 * to fix the number of threads and/or blocks and let the CUDA occupancy
 * calculator determine the unspecified values.
 * The kernel launch is asynchronous.
 */
template <int num_blocks, int num_threads, typename... EnclosedStmts>
using CudaKernelExpAsync =
    CudaKernelExt<cuda_launch<true, num_blocks, num_threads>, EnclosedStmts...>;

/*!
 * A RAJA::kernel statement that launches a CUDA kernel using the
 * CUDA occupancy calculator to determine the optimal number of threads.
 * The kernel launch is synchronous.
 */
template <typename... EnclosedStmts>
using CudaKernelOcc =
    CudaKernelExt<cuda_occ_calc_launch<1024, false>, EnclosedStmts...>;

/*!
 * A RAJA::kernel statement that launches a CUDA kernel using the
 * CUDA occupancy calculator to determine the optimal number of threads.
 * The kernel launch is asynchronous.
 */
template <typename... EnclosedStmts>
using CudaKernelOccAsync =
    CudaKernelExt<cuda_occ_calc_launch<1024, true>, EnclosedStmts...>;

/*!
 * A RAJA::kernel statement that launches a CUDA kernel with a fixed
 * number of threads (specified by num_threads)
 * The kernel launch is synchronous.
 */
template <int num_threads, typename... EnclosedStmts>
using CudaKernelFixed =
    CudaKernelExt<cuda_explicit_launch<false, 0, num_threads>,
                  EnclosedStmts...>;

/*!
 * A RAJA::kernel statement that launches a CUDA kernel with a fixed
 * number of threads (specified by num_threads)
 * The kernel launch is asynchronous.
 */
template <int num_threads, typename... EnclosedStmts>
using CudaKernelFixedAsync =
    CudaKernelExt<cuda_explicit_launch<true, 0, num_threads>, EnclosedStmts...>;

/*!
 * A RAJA::kernel statement that launches a CUDA kernel with 1024 threads
 * Thre kernel launch is synchronous.
 */
template <typename... EnclosedStmts>
using CudaKernel = CudaKernelFixed<1024, EnclosedStmts...>;

/*!
 * A RAJA::kernel statement that launches a CUDA kernel with 1024 threads
 * Thre kernel launch is asynchronous.
 */
template <typename... EnclosedStmts>
using CudaKernelAsync = CudaKernelFixedAsync<1024, EnclosedStmts...>;

}  // namespace statement

namespace internal
{


/*!
 * CUDA global function for launching CudaKernel policies
 */
template <typename Data, typename Exec>
__global__ void CudaKernelLauncher(Data data)
{

  using data_t = camp::decay<Data>;
  data_t private_data = data;

  Exec::exec(private_data, true);
}


/*!
 * CUDA global function for launching CudaKernel policies
 * This is annotated to gaurantee that device code generated
 * can be launched by a kernel with BlockSize number of threads.
 *
 * This launcher is used by the CudaKerelFixed policies.
 */
template <size_t BlockSize, typename Data, typename Exec>
__launch_bounds__(BlockSize, 1) __global__
    void CudaKernelLauncherFixed(Data data)
{

  using data_t = camp::decay<Data>;
  data_t private_data = data;

  // execute the the object
  Exec::exec(private_data, true);
}

/*!
 * Helper class that handles CUDA kernel launching, and computing
 * maximum number of threads/blocks
 */
template<typename LaunchPolicy, typename StmtList, typename Data>
struct CudaLaunchHelper;


/*!
 * Helper class specialization to determine the number of threads and blocks.
 * The user may specify the number of threads and blocks or let one or both be
 * determined at runtime using the CUDA occupancy calculator.
 */
template<bool async0, int num_blocks, int num_threads, typename StmtList, typename Data>
struct CudaLaunchHelper<cuda_launch<async0, num_blocks, num_threads>,StmtList,Data>
{
  using Self = CudaLaunchHelper;

  static constexpr bool async = async0;

  using executor_t = internal::cuda_statement_list_executor_t<StmtList, Data>;

  inline static void recommended_blocks_threads(int shmem_size,
      int &recommended_blocks, int &recommended_threads)
  {
    auto func = internal::CudaKernelLauncher<Data, executor_t>;

    if (num_blocks <= 0) {

      if (num_threads <= 0) {

        //
        // determine blocks at runtime
        // determine threads at runtime
        //
        internal::cuda_occupancy_max_blocks_threads<Self>(
            func, shmem_size, recommended_blocks, recommended_threads);

      } else {

        //
        // determine blocks at runtime
        // threads determined at compile-time
        //
        recommended_threads = num_threads;

        internal::cuda_occupancy_max_blocks<Self, num_threads>(
            func, shmem_size, recommended_blocks);

      }

    } else {

      if (num_threads <= 0) {

        //
        // determine threads at runtime, unsure what use 1024
        // this value may be invalid for kernels with high register pressure
        //
        recommended_threads = 1024;

      } else {

        //
        // threads determined at compile-time
        //
        recommended_threads = num_threads;

      }

      //
      // blocks determined at compile-time
      //
      recommended_blocks = num_blocks;

    }
  }

  inline static void max_threads(int RAJA_UNUSED_ARG(shmem_size), int &max_threads)
  {
    if (num_threads <= 0) {

      //
      // determine threads at runtime, unsure what use 1024
      // this value may be invalid for kernels with high register pressure
      //
      max_threads = 1024;

    } else {

      //
      // threads determined at compile-time
      //
      max_threads = num_threads;

    }
  }

  //
  // actual_threads can be modified in the case where
  // the requested number of theads would cause the
  // max_blocks to be 0
  //
  inline static void max_blocks(int shmem_size,
      int &max_blocks, int &actual_threads)
  {
    auto func = internal::CudaKernelLauncher<Data, executor_t>;

    if (num_blocks <= 0) {

      //
      // determine blocks at runtime
      //
      internal::cuda_occupancy_max_blocks<Self>(
          func, shmem_size, max_blocks, actual_threads);

      if(max_blocks == 0) {

        //
        // actual_threads is too high for the kernel to run
        // successfully due to resource constraints reset to
        // something that is possible
        //
        int min_grid_size, max_block_size;
        // NOTE: min_grid_size is the minimum grid size needed
        //       to achieve the best potential occupancy
        cudaErrchk(cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size, &max_block_size, func, shmem_size));
        actual_threads = max_block_size;
        max_blocks = min_grid_size;
      }


    } else {

      //
      // blocks determined at compile-time
      //
      max_blocks = num_blocks;

    }
  }

  static void launch(Data &&data,
                     internal::LaunchDims launch_dims,
                     size_t shmem,
                     cudaStream_t stream)
  {
    auto func = internal::CudaKernelLauncher<Data, executor_t>;

    void *args[] = {(void*)&data};
    RAJA::cuda::launch((const void*)func, launch_dims.blocks, launch_dims.threads, args, shmem, stream);
  }
};

/*!
 * Helper function that is used to compute either the number of blocks
 * or threads that get launched.
 * It takes the max threads (limit), the requested number (result),
 * and a minimum limit (minimum).
 *
 * The algorithm is greedy (and probably could be improved), and favors
 * maximizing the number of threads (or blocks) in x, y, then z.
 */
inline
cuda_dim_t fitCudaDims(unsigned int limit, cuda_dim_t result, cuda_dim_t minimum = cuda_dim_t()){


  // clamp things to at least 1
  result.x = result.x ? result.x : 1;
  result.y = result.y ? result.y : 1;
  result.z = result.z ? result.z : 1;

  minimum.x = minimum.x ? minimum.x : 1;
  minimum.y = minimum.y ? minimum.y : 1;
  minimum.z = minimum.z ? minimum.z : 1;

  // if we are under the limit, we're done
  if(result.x * result.y * result.z <= limit) return result;

  // Can we reduce z to fit?
  if(result.x * result.y * minimum.z < limit){
    // compute a new z
    result.z = limit / (result.x*result.y);
    return result;
  }
  // we don't fit, so reduce z to it's minimum and continue on to y
  result.z = minimum.z;


  // Can we reduce y to fit?
  if(result.x * minimum.y * result.z < limit){
    // compute a new y
    result.y = limit / (result.x*result.z);
    return result;
  }
  // we don't fit, so reduce y to it's minimum and continue on to x
  result.y = minimum.y;


  // Can we reduce y to fit?
  if(minimum.x * result.y * result.z < limit){
    // compute a new x
    result.x = limit / (result.y*result.z);
    return result;
  }
  // we don't fit, so we'll return the smallest possible thing
  result.x = minimum.x;

  return result;
}


/*!
 * Specialization that launches CUDA kernels for RAJA::kernel from host code
 */
template <typename LaunchConfig, typename... EnclosedStmts>
struct StatementExecutor<
    statement::CudaKernelExt<LaunchConfig, EnclosedStmts...>> {

  using stmt_list_t = StatementList<EnclosedStmts...>;
  using StatementType =
      statement::CudaKernelExt<LaunchConfig, EnclosedStmts...>;

  template <typename Data>
  static inline void exec(Data &&data)
  {

    using data_t = camp::decay<Data>;
    using executor_t = cuda_statement_list_executor_t<stmt_list_t, data_t>;
    using launch_t = CudaLaunchHelper<LaunchConfig, stmt_list_t, data_t>;


    //
    // Compute the requested kernel dimensions
    //
    LaunchDims launch_dims = executor_t::calculateDimensions(data);


    // Only launch kernel if we have something to iterate over
    int num_blocks = launch_dims.num_blocks();
    int num_threads = launch_dims.num_threads();
    if (num_blocks > 0 || num_threads > 0) {

      //
      // Setup shared memory buffers
      //
      int shmem = 0;
      cudaStream_t stream = 0;


      //
      // Compute the recommended physical kernel blocks and threads
      //
      int recommended_blocks, recommended_threads;
      launch_t::recommended_blocks_threads(
          shmem, recommended_blocks, recommended_threads);


      //
      // Compute the MAX physical kernel threads
      //
      int max_threads;
      launch_t::max_threads(shmem, max_threads);


      //
      // Fit the requested threads
      //
      cuda_dim_t fit_threads{0,0,0};

      if ( recommended_threads >= get_size(launch_dims.min_threads) ) {

        fit_threads = fitCudaDims(
            recommended_threads, launch_dims.threads, launch_dims.min_threads);

      }

      //
      // Redo fit with max threads
      //
      if ( recommended_threads < max_threads &&
           get_size(fit_threads) != recommended_threads ) {

        fit_threads = fitCudaDims(
            max_threads, launch_dims.threads, launch_dims.min_threads);

      }

      launch_dims.threads = fit_threads;


      //
      // Compute the MAX physical kernel blocks
      //
      int max_blocks;
      int adjusted_threads = launch_dims.num_threads();
      launch_t::max_blocks(shmem, max_blocks, adjusted_threads);
      if(launch_dims.num_threads() != adjusted_threads) {
        launch_dims.threads = adjusted_threads;
      }

      int use_blocks;

      if ( launch_dims.num_threads() == recommended_threads ) {

        //
        // Fit the requested blocks
        //
        use_blocks = recommended_blocks;

      } else {

        //
        // Fit the max blocks
        //
        use_blocks = max_blocks;

      }

      launch_dims.blocks = fitCudaDims(
          use_blocks, launch_dims.blocks, launch_dims.min_blocks);

      //
      // make sure that we fit
      //
      if(launch_dims.num_blocks() > max_blocks){
        RAJA_ABORT_OR_THROW("RAJA::kernel exceeds max num blocks");
      }
      if(launch_dims.num_threads() > max_threads){
        RAJA_ABORT_OR_THROW("RAJA::kernel exceeds max num threads");
      }

      {
        //
        // Privatize the LoopData, using make_launch_body to setup reductions
        //
        auto cuda_data = RAJA::cuda::make_launch_body(
            launch_dims.blocks, launch_dims.threads, shmem, stream, data);


        //
        // Launch the kernels
        //
        launch_t::launch(std::move(cuda_data), launch_dims, shmem, stream);
      }

      //
      // Synchronize
      //
      if (!launch_t::async) { RAJA::cuda::synchronize(stream); }
    }
  }
};


}  // namespace internal
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
