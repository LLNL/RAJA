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

#ifndef RAJA_policy_cuda_kernel_Lambda_HPP
#define RAJA_policy_cuda_kernel_Lambda_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include <cassert>
#include <climits>

#include "camp/camp.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/pattern/kernel.hpp"
#include "RAJA/pattern/kernel/Lambda.hpp"


namespace RAJA
{
namespace internal
{

template <typename Data, camp::idx_t LoopIndex>
struct CudaStatementExecutor<Data, statement::Lambda<LoopIndex>> {

  static
  inline RAJA_DEVICE void exec(Data &data, bool thread_active)
  {
    // Only execute the lambda if it hasn't been masked off
    if(thread_active){
      invoke_lambda<LoopIndex>(data);
    }
  }


  static
  inline
  LaunchDims calculateDimensions(Data const & RAJA_UNUSED_ARG(data))
  {
    return LaunchDims();
  }
};

//

template <typename Data, camp::idx_t LoopIndex, typename... Args>
struct CudaStatementExecutor<Data, statement::Lambda<LoopIndex, Args...>> {

  static
  inline RAJA_DEVICE void exec(Data &data, bool thread_active)
  {

    //Convert SegList, ParamList into Seg, Param types, and store in a list
    auto targList = parser<camp::list<Args...>>::checkArgs();

    //Create a tuple with the appropriate lambda arguments
    auto argTuple = call_extractor<decltype(targList)>::make_tuple(data);

    //Invoke the lambda with custom arguments
    const int tuple_size = camp::tuple_size<decltype(argTuple)>::value;

    // Only execute the lambda if it hasn't been masked off
    if(thread_active){
      invoke_lambda_with_args<LoopIndex>(data,
                                         argTuple,camp::make_idx_seq_t<tuple_size>{});
    }

  }


  static
  inline
  LaunchDims calculateDimensions(Data const & RAJA_UNUSED_ARG(data))
  {
    return LaunchDims();
  }
};




}  // namespace internal
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
