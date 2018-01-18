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
 * CUDA kernel launch policy where the user specifies the number of physical
 * thread blocks and threads per block.
 */
template <bool async0, int num_blocks, int num_threads>
struct cuda_explicit_launch{

  static constexpr bool async = async0;

  template<typename Func>
  RAJA_INLINE
  static internal::LaunchDim calc_max_physical(Func const &, size_t ){

    return internal::LaunchDim(num_blocks, num_threads);
  }
};


/*!
 * CUDA kernel launch policy where the number of physical blocks and threads
 * are determined by the CUDA occupancy calculator.
 */
template <bool async0>
struct cuda_occ_calc_launch{

  static constexpr bool async = async0;


  template<typename Func>
  RAJA_INLINE
  static internal::LaunchDim calc_max_physical(Func const &func, size_t shmem_size){

    int occ_blocks=-1, occ_threads=-1;

    cudaOccupancyMaxPotentialBlockSize(
        &occ_blocks, &occ_threads, func, shmem_size);


    return internal::LaunchDim(occ_blocks, occ_threads);
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
template <typename... EnclosedStmts>
using CudaKernel = CudaKernelBase<cuda_occ_calc_launch<false>, EnclosedStmts...>;

template <typename... EnclosedStmts>
using CudaKernelAsync = CudaKernelBase<cuda_occ_calc_launch<true>, EnclosedStmts...>;


namespace internal
{


/*!
 * CUDA global function for launching CudaKernel policies
 */
template <typename StmtList, typename Data>
__global__ void CudaKernelLauncher(Data data, long num_logical_blocks)
{
  // Create a struct that hold our current thread allocation
  // this is passed through the meat grinder to properly allocate GPU
  // resources to each executor
  CudaIndexCalc_Terminator index_calc;
  index_calc.num_logical_blocks = num_logical_blocks;
  // use physical block as initial logical block
  index_calc.logical_block = blockIdx.x;

  // Thread privatize the loop data
  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(data);
  auto &private_data = privatizer.get_priv();

  // Iterate through logical blocks
  while(index_calc.logical_block < index_calc.num_logical_blocks){

    // Ensure previous logical block is complete
    // But we don't need to sync on the first logical block
    if(index_calc.logical_block != blockIdx.x){
      __syncthreads();
    }

    // Execute the statement list, using CUDA specific executors
    cuda_execute_statement_list<StmtList>(private_data, index_calc);

    // Increment to the next logical block (grid stride)
    index_calc.logical_block += gridDim.x;
  }
}



/*!
 * Specialization that launches CUDA kernels for nested::forall from host code
 */
template <typename LaunchConfig, typename... EnclosedStmts>
struct StatementExecutor<CudaKernelBase<LaunchConfig, EnclosedStmts...>> {

  using StatementType = CudaKernelBase<LaunchConfig, EnclosedStmts...>;

  template <typename StmtListWrapper>
  void operator()(StmtListWrapper const &wrap)
  {

    int shmem = RAJA::detail::getSharedMemorySize();
    printf("Shared memory size=%d\n", shmem);

    cudaStream_t stream = 0;



    //
    // Compute the MAX physical kernel dimensions
    //

    using data_t = camp::decay<decltype(wrap.data)>;
    LaunchDim max_physical = LaunchConfig::calc_max_physical(CudaKernelLauncher<StatementList<EnclosedStmts...>, data_t>, shmem);


    printf("Physical limits: %ld blocks, %ld threads\n",
        max_physical.blocks, max_physical.threads);







    //
    // Compute the Logical kernel dimensions
    //

    // Privatize the LoopData, using make_launch_body to setup reductions
    auto cuda_data = RAJA::cuda::make_launch_body(max_physical.blocks, max_physical.threads, shmem, stream, wrap.data);
    printf("Data size=%d\n", (int)sizeof(cuda_data));


    // Compute logical dimensions
    using SegmentTuple = decltype(wrap.data.segment_tuple);
    LaunchDim logical_dims =
        cuda_get_statement_list_requested<SegmentTuple, EnclosedStmts...>(wrap.data.segment_tuple, max_physical.blocks, LaunchDim());


    printf("Logical dims: %ld blocks, %ld threads\n",
        logical_dims.blocks, logical_dims.threads);





    //
    // Compute the actual physical kernel dimensions
    //

    LaunchDim launch_dims;
    launch_dims.blocks = std::min(max_physical.blocks, logical_dims.blocks);
    launch_dims.threads = std::min(max_physical.threads, logical_dims.threads);

    printf("Launch dims: %ld blocks, %ld threads\n",
        (long)launch_dims.blocks, (long)launch_dims.threads);





    //
    // Launch the kernels
    //

    CudaKernelLauncher<StatementList<EnclosedStmts...>>
    <<<launch_dims.blocks, launch_dims.threads, shmem, stream>>>(cuda_data, launch_dims.blocks);


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
