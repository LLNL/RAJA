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

#ifndef RAJA_policy_cuda_nested_CudaKernel_HPP
#define RAJA_policy_cuda_nested_CudaKernel_HPP

#include "RAJA/config.hpp"
#include "camp/camp.hpp"
#include "RAJA/pattern/nested.hpp"

#if defined(RAJA_ENABLE_CUDA)

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cassert>
#include <climits>

#include "RAJA/config.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/pattern/nested/For.hpp"
#include "RAJA/pattern/nested/Lambda.hpp"

#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#include "RAJA/policy/cuda/policy.hpp"

#include "RAJA/internal/ForallNPolicy.hpp"
#include "RAJA/internal/LegacyCompatibility.hpp"


namespace RAJA
{
namespace nested
{


/*!
 * CUDA kernel launch policy where the user specifies the number of thread
 * blocks and threads per block.
 */
template <bool async0, int num_blocks, int num_threads>
struct cuda_explicit_launch{

  static constexpr bool async = async0;

  static CudaDim compute_launch_dims(){
    CudaDim d;

    d.num_blocks.x = num_blocks;
    d.num_blocks.y = 1;
    d.num_blocks.z = 1;

    d.num_threads.x = num_threads;
    d.num_threads.y = 1;
    d.num_threads.z = 1;


    return d;
  }
};


/*!
 * CUDA kernel launch policy where the user specifies the number of threads
 * per block, and the number of blocks per Streaming MultiProcessor.
 *
 * A value of 0 for num_blocks_per_sm (default) uses the device properties
 * to compute how many blocks will will fit on each SM to maximize occupancy.
 */
template <bool async0, int num_threads, int num_blocks_per_sm=0>
struct cuda_block_per_sm_launch{

  static constexpr bool async = async0;

  static CudaDim compute_launch_dims(){
    CudaDim d;

    // Get current device's properties
    int cur_device = -1;
    cudaGetDevice(&cur_device);
    cudaDeviceProp dev_props;
    cudaGetDeviceProperties(&dev_props, cur_device);

    // Compute number of blocks
    int num_sm = dev_props.multiProcessorCount;
    int num_blocks = num_sm * num_blocks_per_sm;

    if(num_blocks_per_sm == 0){
      int num_threads_per_sm = dev_props.maxThreadsPerMultiProcessor;
      int blocks_per_sm = num_threads_per_sm / num_threads;
      num_blocks = blocks_per_sm * num_sm;

      // limit it to 8 blocks/sm
      // TODO: is there a way to compute max resident blocks/sm?!?!?
      if(num_blocks > num_sm*8){
        num_blocks = num_sm*8;
      }
    }

    d.num_blocks.x = num_blocks;
    d.num_blocks.y = 1;
    d.num_blocks.z = 1;

    d.num_threads.x = num_threads;
    d.num_threads.y = 1;
    d.num_threads.z = 1;


    return d;
  }
};


/*!
 * A nested::forall statement that launches a CUDA kernel.
 *
 *
 */
template <typename LaunchConfig, typename... EnclosedStmts>
struct CudaKernelBase : public internal::Statement<cuda_exec<0>, EnclosedStmts...>{
};


/*!
 * A nested::forall statement that launches a CUDA kernel.
 *
 *
 */
template <int num_threads, typename... EnclosedStmts>
using CudaKernel = CudaKernelBase<cuda_block_per_sm_launch<false, num_threads>, EnclosedStmts...>;

template <int num_threads, typename... EnclosedStmts>
using CudaKernelAsync = CudaKernelBase<cuda_block_per_sm_launch<true, num_threads>, EnclosedStmts...>;


namespace internal
{


/*!
 * CUDA global function for launching CudaKernel policies
 */
template <typename StmtList, typename Data>
__global__ void CudaKernelLauncher(StmtList st, Data data)
{
  // Create a struct that hold our current thread allocation
  // this is passed through the meat grinder to properly allocate GPU
  // resources to each executor
  CudaExecInfo exec_info;

  // Thread privatize the loop data
  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(data);
  auto &private_data = privatizer.get_priv();

  // Execute the statement list, using CUDA specific executors
  CudaStatementListWrapper<StmtList, Data> cuda_wrapper(st, private_data);
  cuda_wrapper(exec_info);
}



/*!
 * Specialization that launches CUDA kernels for nested::forall from host code
 */
template <typename LaunchConfig, typename... EnclosedStmts>
struct StatementExecutor<CudaKernelBase<LaunchConfig, EnclosedStmts...>> {

  using StatementType = CudaKernelBase<LaunchConfig, EnclosedStmts...>;

  template <typename StmtListWrapper>
  void operator()(StatementType const &fp, StmtListWrapper const &wrap)
  {

    using data_type = camp::decay<typename StmtListWrapper::data_type>;
    using stmt_list_type = camp::decay<typename StmtListWrapper::statement_list_type>;

    // Use the LaunchConfig type to compute how many threads and blocks to use
    CudaDim dims = LaunchConfig::compute_launch_dims();

    cudaStream_t stream = 0;
    int shmem = RAJA::detail::getSharedMemorySize();
    printf("Shared memory size=%d\n", shmem);
    dims.print();

    // Launch, using make_launch_body to correctly setup reductions
    CudaKernelLauncher<<<dims.num_blocks, dims.num_threads, shmem, stream>>>(
        wrap.statement_list,
        RAJA::cuda::make_launch_body(dims.num_blocks.x, dims.num_threads.x, shmem, stream, wrap.data ));


    // Check for errors
    RAJA::cuda::peekAtLastError();

    RAJA::cuda::launch(stream);

    if (!LaunchConfig::async){
      RAJA::cuda::synchronize(stream);
    }
  }
};



}  // namespace internal
}  // namespace nested
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
