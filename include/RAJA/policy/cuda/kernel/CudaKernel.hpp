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
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
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
 * CUDA kernel launch policy where the user specifies the number of physical
 * thread blocks and threads per block.
 */
template <bool async0, int num_blocks, int num_threads>
struct cuda_explicit_launch {};


/*!
 * CUDA kernel launch policy where the number of physical blocks and threads
 * are determined by the CUDA occupancy calculator.
 */
template <int num_threads0, bool async0>
struct cuda_occ_calc_launch {};


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
 * Thre kernel launch is asynchronous.
 */
template <typename... EnclosedStmts>
using CudaKernelOccAsync =
    CudaKernelExt<cuda_occ_calc_launch<1024, true>, EnclosedStmts...>;

/*!
 * A RAJA::kernel statement that launches a CUDA kernel with a fixed
 * number of threads (specified by num_threads)
 * Thre kernel launch is synchronous.
 */
template <int num_threads, typename... EnclosedStmts>
using CudaKernelFixed =
    CudaKernelExt<cuda_explicit_launch<false, 0, num_threads>,
                  EnclosedStmts...>;

/*!
 * A RAJA::kernel statement that launches a CUDA kernel with a fixed
 * number of threads (specified by num_threads)
 * Thre kernel launch is asynchronous.
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
 * Helper class specialization to use the CUDA occupancy calculator to
 * determine the number of threads and blocks
 */
template<int num_threads, bool async0, typename StmtList, typename Data>
struct CudaLaunchHelper<cuda_occ_calc_launch<num_threads, async0>,StmtList,Data>
{
  static constexpr bool async = async0;

  using executor_t = internal::cuda_statement_list_executor_t<StmtList, Data>;

  inline static void max_blocks_threads(int shmem_size,
      int &max_blocks, int &max_threads)
  {

    auto func = internal::CudaKernelLauncher<Data, executor_t>;

    cudaOccupancyMaxPotentialBlockSize(&max_blocks,
                                       &max_threads,
                                       func,
                                       shmem_size);

  }

  static void launch(Data const &data,
                     internal::LaunchDims launch_dims,
                     size_t shmem,
                     cudaStream_t stream)
  {

    auto func = internal::CudaKernelLauncher<Data, executor_t>;

    func<<<launch_dims.blocks, launch_dims.threads, shmem, stream>>>(data);
  }
};



/*!
 * Helper class specialization to use the CUDA device properties and a user
 * specified number of threads to compute the number of blocks/threads
 */
template<bool async0, int num_blocks, int num_threads, typename StmtList, typename Data>
struct CudaLaunchHelper<cuda_explicit_launch<async0, num_blocks, num_threads>,StmtList,Data>
{
  static constexpr bool async = async0;

  using executor_t = internal::cuda_statement_list_executor_t<StmtList, Data>;

  inline static void max_blocks_threads(int shmem_size,
      int &max_blocks, int &max_threads)
  {

    max_blocks = num_blocks;
    max_threads = num_threads;

    // Use maximum number of blocks if user didn't specify
    if (num_blocks <= 0) {
      max_blocks = RAJA::cuda::internal::getMaxBlocks();
    }

  }

  static void launch(Data const &data,
                     internal::LaunchDims launch_dims,
                     size_t shmem,
                     cudaStream_t stream)
  {

    auto func = internal::CudaKernelLauncherFixed<num_threads,Data, executor_t>;

    func<<<launch_dims.blocks, launch_dims.threads, shmem, stream>>>(data);
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
cuda_dim_t fitCudaDims(int limit, cuda_dim_t result, cuda_dim_t minimum = cuda_dim_t()){


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
    // Setup shared memory buffers
    //
    int shmem = 0;
    cudaStream_t stream = 0;


    //
    // Compute the MAX physical kernel dimensions
    //
    int max_blocks, max_threads;
    launch_t::max_blocks_threads(shmem, max_blocks, max_threads);


    //
    // Privatize the LoopData, using make_launch_body to setup reductions
    //
    auto cuda_data = RAJA::cuda::make_launch_body(
        max_blocks, max_threads, shmem, stream, data);


    //
    // Compute the requested kernel dimensions
    //
    LaunchDims launch_dims = executor_t::calculateDimensions(data);


    // Only launch kernel if we have something to iterate over
    int num_blocks = launch_dims.num_blocks();
    int num_threads = launch_dims.num_threads();
    if (num_blocks > 0 || num_threads > 0) {

      //
      // Fit the requested threads an blocks
      //
      launch_dims.blocks = fitCudaDims(max_blocks, launch_dims.blocks);
      launch_dims.threads = fitCudaDims(max_threads, launch_dims.threads, launch_dims.min_threads);

      // make sure that we fit
      if(launch_dims.num_blocks() > max_blocks){
        RAJA_ABORT_OR_THROW("RAJA::kernel exceeds max num blocks");
      }
      if(launch_dims.num_threads() > max_threads){
        RAJA_ABORT_OR_THROW("RAJA::kernel exceeds max num threads");
      }

      //
      // Launch the kernels
      //
      launch_t::launch(cuda_data, launch_dims, shmem, stream);


      //
      // Check for errors
      //
      RAJA::cuda::peekAtLastError();


      //
      // Synchronize
      //
      RAJA::cuda::launch(stream);

      if (!launch_t::async) {
        RAJA::cuda::synchronize(stream);
      }
    }
  }
};


}  // namespace internal
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
