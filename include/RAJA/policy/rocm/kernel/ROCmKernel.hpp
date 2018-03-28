#include "hip/hip_runtime.h"
/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run forallN
 *          traversals on GPU with ROCM.
 *
 ******************************************************************************
 */

#ifndef RAJA_policy_rocm_kernel_ROCmKernel_HPP
#define RAJA_policy_rocm_kernel_ROCmKernel_HPP

#include "RAJA/config.hpp"
#include "RAJA/pattern/kernel.hpp"
#include "camp/camp.hpp"

#if defined(RAJA_ENABLE_ROCM)

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

#include "RAJA/policy/rocm/MemUtils_ROCM.hpp"
#include "RAJA/policy/rocm/policy.hpp"

#include "RAJA/internal/ForallNPolicy.hpp"
#include "RAJA/internal/LegacyCompatibility.hpp"


namespace RAJA
{


/*!
 * ROCM kernel launch policy where the user specifies the number of physical
 * thread blocks and threads per block.
 */
template <bool async0, int num_blocks, int num_threads>
struct rocm_explicit_launch {

  static constexpr bool async = async0;

  template <typename Func>
  RAJA_INLINE static internal::LaunchDim calc_max_physical(Func const &, int)
  {

    return internal::LaunchDim(num_blocks, num_threads);
  }
};


/*!
 * ROCM kernel launch policy where the number of physical blocks and threads
 * are determined by the ROCM occupancy calculator.
 */
template <int num_threads0, bool async0>
struct rocm_occ_calc_launch {

  static constexpr bool async = async0;

  static constexpr int num_threads = num_threads0;

  template <typename Func>
  RAJA_INLINE static internal::LaunchDim calc_max_physical(Func const &func,
                                                           int shmem_size)
  {

    int occ_blocks = -1, occ_threads = -1;

    rocmOccupancyMaxPotentialBlockSize(&occ_blocks,
                                       &occ_threads,
                                       func,
                                       shmem_size);

    return internal::LaunchDim(occ_blocks, occ_threads);
  }
};

namespace statement
{

/*!
 * A kernel::forall statement that launches a ROCM kernel.
 *
 *
 */
template <typename LaunchConfig, typename... EnclosedStmts>
struct ROCmKernelExt
    : public internal::Statement<rocm_exec<0>, EnclosedStmts...> {
};


/*!
 * A kernel::forall statement that launches a ROCM kernel.
 *
 *
 */
template <typename... EnclosedStmts>
using ROCmKernel =
    ROCmKernelExt<rocm_occ_calc_launch<1024, false>, EnclosedStmts...>;

template <typename... EnclosedStmts>
using ROCmKernelAsync =
    ROCmKernelExt<rocm_occ_calc_launch<1024, true>, EnclosedStmts...>;

}  // namespace statement

namespace internal
{


/*!
 * ROCM global function for launching ROCmKernel policies
 */
template <typename StmtList, typename Data>
//__launch_bounds__(1024, 112)
__global__ void ROCmKernelLauncher(Data data, int num_logical_blocks)
{
  using data_t = camp::decay<Data>;
  data_t private_data = data;

  // Instantiate an executor object
  using executor_t = rocm_statement_list_executor_t<StmtList, Data>;
  executor_t executor;

  // execute the the object
  executor.exec(private_data, num_logical_blocks, -1);
}


/*!
 * Specialization that launches ROCM kernels for kernel::forall from host code
 */
template <typename LaunchConfig, typename... EnclosedStmts>
struct StatementExecutor<statement::ROCmKernelExt<LaunchConfig,
                                                  EnclosedStmts...>> {

  using stmt_list_t = StatementList<EnclosedStmts...>;
  using StatementType =
      statement::ROCmKernelExt<LaunchConfig, EnclosedStmts...>;

  template <typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {

    int shmem = (int)RAJA::internal::shmem_setup_buffers(data.param_tuple);
    //    printf("Shared memory size=%d\n", (int)shmem);

    hipStream_t stream = 0;


    //
    // Compute the MAX physical kernel dimensions
    //

    using data_t = camp::decay<Data>;
    LaunchDim max_physical = LaunchConfig::calc_max_physical(
        ROCmKernelLauncher<StatementList<EnclosedStmts...>, data_t>, shmem);

    //    printf("Physical limits: %d blocks, %d threads\n",
    //        (int)max_physical.blocks, (int)max_physical.threads);


    //
    // Compute the Logical kernel dimensions
    //

    // Privatize the LoopData, using make_launch_body to setup reductions
    auto rocm_data = RAJA::rocm::make_launch_body(
        max_physical.blocks, max_physical.threads, shmem, stream, data);
    //    printf("Data size=%d\n", (int)sizeof(rocm_data));


    // Compute logical dimensions
    using SegmentTuple = decltype(data.segment_tuple);

    // Instantiate an executor object
    using executor_t = rocm_statement_list_executor_t<stmt_list_t, data_t>;
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
    hipLaunchKernelGGL((ROCmKernelLauncher<StatementList<EnclosedStmts...>>), dim3(launch_dims.blocks), dim3(launch_dims.threads), shmem, stream, 
            rocm_data, logical_dims.blocks);


    // Check for errors
    RAJA::rocm::peekAtLastError();

    RAJA::rocm::launch(stream);

    if (!LaunchConfig::async) {
      RAJA::rocm::synchronize(stream);
    }
  }
};


}  // namespace internal
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_ROCM guard

#endif  // closing endif for header file include guard
