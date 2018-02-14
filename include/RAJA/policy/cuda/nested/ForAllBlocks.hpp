#ifndef RAJA_policy_cuda_nested_ForAllBlocks_HPP
#define RAJA_policy_cuda_nested_ForAllBlocks_HPP

#include "RAJA/config.hpp"
#include "RAJA/policy/cuda/nested/internal.hpp"



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


namespace RAJA
{
namespace nested
{


/*!
 * A nested::forall statement that allows threaded execution over all physical
 * blocks, regardless of work distribution.
 *
 * This is useful for setting up shared memory regions.
 *
 */
template <typename... EnclosedStmts>
struct ForAllBlocks : public internal::Statement<camp::nil, EnclosedStmts...>
{};



namespace internal{



/*
 * Executor for sequential loops inside of a Cuda Kernel.
 *
 * This is specialized since it need to execute the loop immediately.
 */
template <typename... EnclosedStmts, typename IndexCalc>
struct CudaStatementExecutor<ForAllBlocks<EnclosedStmts...>, IndexCalc> {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  template <typename Data>
  static
  inline
  RAJA_DEVICE
  void exec(Data &data, int logical_block)
  {
    // execute enclosed statements
    cuda_execute_statement_list<stmt_list_t, IndexCalc>(data, 0);
  }


  template<typename Data>
  static
  RAJA_INLINE
  LaunchDim calculateDimensions(Data const &data, LaunchDim const &max_physical){

    // Return launch dimensions of enclosed statements
    return cuda_calcdims_statement_list<stmt_list_t, IndexCalc>(data, max_physical);
  }


};




} // namespace internal
}  // end namespace nested
}  // end namespace RAJA



#endif /* RAJA_pattern_nested_HPP */
