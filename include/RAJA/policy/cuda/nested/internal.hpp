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

#ifndef RAJA_policy_cuda_nested_internal_HPP
#define RAJA_policy_cuda_nested_internal_HPP

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
namespace internal
{


struct CudaExecInfo {
  short thread_id;
  short threads_left;

  RAJA_DEVICE
  constexpr
  CudaExecInfo() :
    thread_id(threadIdx.x),
    threads_left(blockDim.x)
  {}
};


template <typename Policy>
struct CudaStatementExecutor{};

template <camp::idx_t idx, camp::idx_t N, typename StmtList>
struct CudaStatementListExecutor;



template<typename StmtList, typename Data>
RAJA_DEVICE
RAJA_INLINE
void cuda_execute_statement_list(Data &data, CudaExecInfo &exec_info){

  CudaStatementListExecutor<0, StmtList::size, StmtList>::exec(data, exec_info);

}


template <typename StmtList, typename Data>
struct CudaStatementListWrapper {

  RAJA_INLINE
  RAJA_DEVICE
  void operator()(Data &data, CudaExecInfo &exec_info) const
  {
    cuda_execute_statement_list<StmtList>(data, exec_info);
  }
};


// Create a wrapper for this policy
template<typename PolicyT, typename Data>
RAJA_INLINE
RAJA_DEVICE
constexpr
auto cuda_make_statement_list_wrapper(Data & data) ->
  CudaStatementListWrapper<PolicyT, camp::decay<Data>>
{
  return CudaStatementListWrapper<PolicyT, camp::decay<Data>>();
}


template <camp::idx_t statement_index, camp::idx_t num_statements, typename StmtList>
struct CudaStatementListExecutor{

  template<typename Data>
  static
  RAJA_DEVICE
  void exec(Data &data, CudaExecInfo &exec_info){

    // Get the statement we're going to execute
    using statement = camp::at_v<StmtList, statement_index>;

    // Create a wrapper for enclosed statements within statement
    using eclosed_statements_t = typename statement::enclosed_statements_t;
    auto enclosed_wrapper = cuda_make_statement_list_wrapper<eclosed_statements_t>(data);

    // Execute this statement
    CudaStatementExecutor<statement>::exec(enclosed_wrapper, data, exec_info);

    // call our next statement
    CudaStatementListExecutor<statement_index+1, num_statements, StmtList>::exec(data, exec_info);
  }
};


/*
 * termination case, a NOP.
 */

template <camp::idx_t num_statements, typename StmtList>
struct CudaStatementListExecutor<num_statements,num_statements, StmtList> {

  template<typename Data>
  static
  RAJA_INLINE
  RAJA_DEVICE
  void exec(Data &, CudaExecInfo &) {}

};




}  // namespace internal
}  // namespace nested
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
