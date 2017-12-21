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
 * A nested::forall statement that launches a CUDA kernel.
 *
 *
 */
template <int num_blocks, int num_threads, typename... EnclosedStmts>
struct CudaKernel : public internal::Statement<cuda_exec<0>, EnclosedStmts...>{

};


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
//  using data_type = camp::decay<Data>;
//#if 0
//  // Use shared memory for privatized data (no chance of registers here)
//  __shared__ char private_data_raw[sizeof(data_type)];
//  data_type &private_data = *reinterpret_cast<data_type*>(&private_data_raw[0]);
//  memcpy(&private_data, &data, sizeof(data_type));
//#else
//  // Use global memory (or possibly registers) for privatized data
//  //data_type private_data{data};
//#endif


  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(data);
  auto &private_data = privatizer.get_priv();


  // Execute the statement list, using CUDA specific executors
  CudaStatementListWrapper<StmtList, Data> cuda_wrapper(st, private_data);
  cuda_wrapper(exec_info);
}


template <int num_blocks, int num_threads, typename... EnclosedStmts>
struct StatementExecutor<CudaKernel<num_blocks, num_threads, EnclosedStmts...>> {

  using StatementType = CudaKernel<num_blocks, num_threads, EnclosedStmts...>;

  template <typename StmtListWrapper>
  void operator()(StatementType const &fp, StmtListWrapper const &wrap)
  {

    printf("LAUNCH KERNEL with %d blocks and  %d threads\n", num_blocks, num_threads);

    using data_type = camp::decay<typename StmtListWrapper::data_type>;
    //data_type private_data = wrap.data;

    using stmt_list_type = camp::decay<typename StmtListWrapper::statement_list_type>;
    stmt_list_type private_stmt_list = wrap.statement_list;

    cudaStream_t stream = 0;
    int shmem = 0;

    // setup reducers
    //printf("creating private_data\n");
    //using loop_data_t = decltype(wrap.data);
    //data_type private_data = RAJA::cuda::make_launch_body(num_blocks, num_threads, shmem, stream, std::forward<loop_data_t>(wrap.data));

//    printf("launching kernel\n");
    // Launch kernel
//    CudaKernelLauncher<<<num_blocks, num_threads, shmem, stream>>>(private_stmt_list, std::move(private_data));
    CudaKernelLauncher<<<num_blocks, num_threads, shmem, stream>>>(
        private_stmt_list,
        RAJA::cuda::make_launch_body(num_blocks, num_threads, shmem, stream, wrap.data ));

    //printf("kernel complete, private_data=%p\n", &private_data);
//    printf("kernel complete\n");

    // Check for errors
    RAJA::cuda::peekAtLastError();

    RAJA::cuda::launch(stream);
    //if (!Async)
    RAJA::cuda::synchronize(stream);

  }
};



}  // namespace internal
}  // namespace nested
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
