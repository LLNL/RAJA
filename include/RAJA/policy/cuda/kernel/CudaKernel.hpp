/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run forallN
 *          traversals on GPU with CUDA.
 *
 ******************************************************************************
 */

#ifndef RAJA_policy_cuda_kernel_CudaKernel_HPP
#define RAJA_policy_cuda_kernel_CudaKernel_HPP

#include "RAJA/config.hpp"
#include "RAJA/pattern/kernel.hpp"
#include "camp/camp.hpp"

#if defined(RAJA_ENABLE_CUDA)

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
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


#include <cassert>
#include <climits>

#include "RAJA/config.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/pattern/kernel/For.hpp"
#include "RAJA/pattern/kernel/Lambda.hpp"

#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#include "RAJA/policy/cuda/policy.hpp"

#include "RAJA/internal/ForallNPolicy.hpp"
#include "RAJA/internal/LegacyCompatibility.hpp"


namespace RAJA
{


/*!
 * CUDA kernel launch policy where the user specifies the number of physical
 * thread blocks and threads per block.
 */
template <bool async0, int num_blocks, int num_threads>
struct cuda_explicit_launch {

  static constexpr bool async = async0;

  template <typename Func>
  RAJA_INLINE static internal::LaunchDim calc_max_physical(Func const &, int)
  {

    return internal::LaunchDim(num_blocks, num_threads);
  }
};


/*!
 * CUDA kernel launch policy where the number of physical blocks and threads
 * are determined by the CUDA occupancy calculator.
 */
template <int num_threads0, bool async0>
struct cuda_occ_calc_launch {

  static constexpr bool async = async0;

  static constexpr int num_threads = num_threads0;

  template <typename Func>
  RAJA_INLINE static internal::LaunchDim calc_max_physical(Func const &func,
                                                           int shmem_size)
  {

    int occ_blocks = -1, occ_threads = -1;

    cudaOccupancyMaxPotentialBlockSize(&occ_blocks,
                                       &occ_threads,
                                       func,
                                       shmem_size);

    return internal::LaunchDim(occ_blocks, occ_threads);
  }
};

namespace statement
{

/*!
 * A kernel::forall statement that launches a CUDA kernel.
 *
 *
 */
template <typename LaunchConfig, typename... EnclosedStmts>
struct CudaKernelExt
    : public internal::Statement<cuda_exec<0>, EnclosedStmts...> {
};


/*!
 * A kernel::forall statement that launches a CUDA kernel.
 *
 *
 */
template <typename... EnclosedStmts>
using CudaKernel =
    CudaKernelExt<cuda_occ_calc_launch<1024, false>, EnclosedStmts...>;

template <typename... EnclosedStmts>
using CudaKernelAsync =
    CudaKernelExt<cuda_occ_calc_launch<1024, true>, EnclosedStmts...>;

}  // namespace statement

namespace internal
{


/*!
 * CUDA global function for launching CudaKernel policies
 */
template <typename StmtList, typename Data>
//__launch_bounds__(1024, 112)
__global__ void CudaKernelLauncher(Data data, int num_logical_blocks)
{
  using data_t = camp::decay<Data>;
  data_t private_data = data;

  // Instantiate an executor object
  using executor_t = cuda_statement_list_executor_t<StmtList, Data>;
  executor_t executor;

  // execute the the object
  executor.exec(private_data, num_logical_blocks, -1);
}


/*!
 * Specialization that launches CUDA kernels for kernel::forall from host code
 */
template <typename LaunchConfig, typename... EnclosedStmts>
struct StatementExecutor<statement::CudaKernelExt<LaunchConfig,
                                                  EnclosedStmts...>> {

  using stmt_list_t = StatementList<EnclosedStmts...>;
  using StatementType =
      statement::CudaKernelExt<LaunchConfig, EnclosedStmts...>;

  template <typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {

    int shmem = (int)RAJA::internal::shmem_setup_buffers(data.param_tuple);
    //    printf("Shared memory size=%d\n", (int)shmem);

    cudaStream_t stream = 0;


    //
    // Compute the MAX physical kernel dimensions
    //

    using data_t = camp::decay<Data>;
    LaunchDim max_physical = LaunchConfig::calc_max_physical(
        CudaKernelLauncher<StatementList<EnclosedStmts...>, data_t>, shmem);

    //    printf("Physical limits: %d blocks, %d threads\n",
    //        (int)max_physical.blocks, (int)max_physical.threads);


    //
    // Compute the Logical kernel dimensions
    //

    // Privatize the LoopData, using make_launch_body to setup reductions
    auto cuda_data = RAJA::cuda::make_launch_body(
        max_physical.blocks, max_physical.threads, shmem, stream, data);
    //    printf("Data size=%d\n", (int)sizeof(cuda_data));


    // Compute logical dimensions
    using SegmentTuple = decltype(data.segment_tuple);

    // Instantiate an executor object
    using executor_t = cuda_statement_list_executor_t<stmt_list_t, data_t>;
    executor_t executor;

    // Compute logical dimensions
    LaunchDim logical_dims = executor.calculateDimensions(data, max_physical);


    //    printf("Logical dims: %d blocks, %d threads\n",
    //        (int)logical_dims.blocks, (int)logical_dims.threads);


    //
    // Compute the actual physical kernel dimensions
    //

    LaunchDim launch_dims;
    launch_dims.blocks = std::min(max_physical.blocks, logical_dims.blocks);
    launch_dims.threads = std::min(max_physical.threads, logical_dims.threads);

    //    printf("Launch dims: %d blocks, %d threads\n",
    //        (int)launch_dims.blocks, (int)launch_dims.threads);


    //
    // Launch the kernels
    //
    CudaKernelLauncher<StatementList<EnclosedStmts...>>
        <<<launch_dims.blocks, launch_dims.threads, shmem, stream>>>(
            cuda_data, logical_dims.blocks);


    // Check for errors
    RAJA::cuda::peekAtLastError();

    RAJA::cuda::launch(stream);

    if (!LaunchConfig::async) {
      RAJA::cuda::synchronize(stream);
    }
  }
};


}  // namespace internal
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
