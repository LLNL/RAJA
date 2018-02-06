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

#ifndef RAJA_policy_cuda_nested_Lambda_HPP
#define RAJA_policy_cuda_nested_Lambda_HPP

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

#include "RAJA/pattern/nested/Lambda.hpp"


namespace RAJA
{
namespace nested
{


/*!
 * A nested::forall statement that executes a lambda function with cuda shared
 * memory support.
 *
 * The lambda is specified by it's index, which is defined by the order in
 * which it was specified in the call to nested::forall.
 *
 * for example:
 * RAJA::nested::forall(pol{}, make_tuple{s0, s1, s2}, lambda0, lambda1);
 *
 */
template <camp::idx_t BodyIdx>
struct ShmemLambda : internal::Statement<camp::nil> {
  const static camp::idx_t loop_body_index = BodyIdx;
};


namespace internal
{

template <camp::idx_t LoopIndex, typename IndexCalc>
struct CudaStatementExecutor<Lambda<LoopIndex>, IndexCalc>{

  template <typename Data>
  static
  inline
  __device__
  void exec(Data &data, int logical_block)
  {
    // Get physical parameters
    LaunchDim max_physical(gridDim.x, blockDim.x);

    // Compute logical dimensions
    IndexCalc index_calc(data.segment_tuple, max_physical);
    int num_logical_threads = index_calc.numLogicalThreads();

    // Loop over logical threads in this block
    int logical_thread = threadIdx.x;
    while(logical_thread < num_logical_threads){

      // compute indices
      bool in_bounds = index_calc.assignIndices(data, logical_block, logical_thread);

      // call the user defined function, if the computed index in in bounds
      if(in_bounds){
        invoke_lambda<LoopIndex>(data);
      }

      // increment to next block-stride logical thread
      logical_thread += blockDim.x;
    }

  }


  template<typename Data>
  static
  RAJA_INLINE
  LaunchDim calculateDimensions(Data const &data, LaunchDim const &max_physical){

    IndexCalc index_calc(data.segment_tuple, max_physical);
    return index_calc.computeLogicalDims();

  }

};



template <camp::idx_t LoopIndex, typename IndexCalc>
struct CudaStatementExecutor<ShmemLambda<LoopIndex>, IndexCalc>{

  template <typename Data>
  static
  inline
  __device__
  void exec(Data &data, int logical_block)
  {
    // Get physical parameters
    LaunchDim max_physical(gridDim.x, blockDim.x);

    // Compute logical dimensions
    IndexCalc index_calc(data.segment_tuple, max_physical);
    int num_logical_threads = index_calc.numLogicalThreads();

    // Loop over logical threads in this block
    int logical_thread = threadIdx.x;

    // Divine the type of the index tuple in wrap.data
    using loop_data_t = camp::decay<Data>;
    using index_tuple_t = camp::decay<typename loop_data_t::index_tuple_t>;

    // make sure all threads are done with current window
    __syncthreads();

    // Grab a pointer to the shmem window tuple.  We are assuming that this
    // is the first thing in the dynamic shared memory
    if(logical_thread == 0){
//      if(blockIdx.x==0){
//        printf("logical_block=%d\n", logical_block);
//      }

      // compute starting indices
      index_calc.assignIndices(data, logical_block, 0);

      // Grab shmem window pointer
      extern __shared__ char my_ptr[];
      index_tuple_t *shmem_window = reinterpret_cast<index_tuple_t *>(&my_ptr[0]);

      // Set the shared memory tuple with the beginning of our segments
      *shmem_window = data.index_tuple;
    }

    // make sure we're all synchronized, so they all see the same window
    __syncthreads();

    // Thread privatize, triggering Shmem objects to grab updated window info
    //auto private_data = privatize_bodies(data);


    while(logical_thread < num_logical_threads){

      // compute indices
      bool in_bounds = index_calc.assignIndices(data, logical_block, logical_thread);

      // call the user defined function, if the computed index in in bounds
      if(in_bounds){
        invoke_lambda<LoopIndex>(data);
      }

      // increment to next block-stride logical thread
      logical_thread += blockDim.x;
    }

  }


  template<typename Data>
  static
  RAJA_INLINE
  LaunchDim calculateDimensions(Data const &data, LaunchDim const &max_physical){

    IndexCalc index_calc(data.segment_tuple, max_physical);
    return index_calc.computeLogicalDims();

  }

};




}  // namespace internal
}  // namespace nested
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
