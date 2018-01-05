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

#ifndef RAJA_policy_cuda_nested_Collapse_HPP
#define RAJA_policy_cuda_nested_Collapse_HPP

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

#include "RAJA/policy/cuda/nested/For.hpp"
#include "RAJA/pattern/nested/Collapse.hpp"
#include "RAJA/util/Layout.hpp"


namespace RAJA
{
namespace nested
{
namespace internal
{


/*
 * This allows us to pass our segment and index tuple from LoopData into
 * the Layout::toIndices() function, and do the proper index calculations
 */
template<camp::idx_t idx, typename segment_tuple_t, typename index_tuple_t>
struct IndexAssigner{

  using Self = IndexAssigner<idx, segment_tuple_t, index_tuple_t>;

  segment_tuple_t &segments;
  index_tuple_t   &indices;

  template<typename T>
  RAJA_INLINE
  RAJA_DEVICE Self const &operator=(T value) const{
    // Compute actual offset using begin() iterator of segment
    auto offset = *(camp::get<idx>(segments).begin() + value);

    // Assign offset into index tuple
    camp::get<idx>(indices) = offset;

    // nobody wants this
    return *this;
  }
};


template<camp::idx_t idx, typename segment_tuple_t, typename index_tuple_t>
RAJA_INLINE
RAJA_DEVICE
auto make_index_assigner(segment_tuple_t &s, index_tuple_t &i) ->
IndexAssigner<idx, segment_tuple_t, index_tuple_t>
{
  return IndexAssigner<idx, segment_tuple_t, index_tuple_t>{s, i};
}

/*
 * Collapses multiple segments iteration space, and distributes them over threads.
 *
 * No work sharing between blocks is performed
 */
template <camp::idx_t ... Args, typename... EnclosedStmts>
struct CudaStatementExecutor<Collapse<cuda_thread_exec, ArgList<Args...>, EnclosedStmts...>> {

  static constexpr size_t num_dims = sizeof...(Args);

  template <typename WrappedBody, typename Data>
  static
  RAJA_DEVICE
  void exec(WrappedBody const &wrap, Data &data, CudaExecInfo &exec_info)
  {
    // Create a Layout of all of our loop dimensions that we're collapsing
    RAJA::Layout<num_dims> layout( (camp::get<Args>(data.segment_tuple).end() -
        camp::get<Args>(data.segment_tuple).begin()) ...);

    // get total iteration size
    ptrdiff_t len = layout.size();

    // How many batches of threads do we need?
    int num_batches = len / exec_info.threads_left;
    if(num_batches*exec_info.threads_left < len){
      num_batches++;
    }

    // compute our starting index
    int i = exec_info.thread_id;

    for(int batch = 0;batch < num_batches;++ batch){

      if(i < len){
        // Compute indices from layout, and assign them to our index tuple
        //layout.toIndices(i, camp::get<Args>(data.index_tuple)...);
        layout.toIndices(i, make_index_assigner<Args>(data.segment_tuple, data.index_tuple)...);


        // invoke our enclosed statement list
        wrap(data, exec_info);
      }

      i += exec_info.threads_left;
    }

  }
};





/*
 * Collapses multiple segments iteration space, and distributes them over
 * all of the blocks and threads.
 */
template <camp::idx_t ... Args, typename... EnclosedStmts>
struct CudaStatementExecutor<Collapse<cuda_block_thread_exec, ArgList<Args...>, EnclosedStmts...>> {

  static constexpr size_t num_dims = sizeof...(Args);

  template <typename WrappedBody, typename Data>
  static
  RAJA_DEVICE
  void exec(WrappedBody const &wrap, Data &data, CudaExecInfo &exec_info)
  {
    // Create a Layout of all of our loop dimensions that we're collapsing
    RAJA::Layout<num_dims> layout( (camp::get<Args>(data.segment_tuple).end() -
        camp::get<Args>(data.segment_tuple).begin()) ...);

    // get total iteration size
    ptrdiff_t total_len = layout.size();

    // compute our block's slice of work
    int num_blocks = gridDim.x;
    auto block_len = total_len / num_blocks;
    if(block_len*num_blocks < total_len){
      block_len ++;
    }
    auto block_begin = block_len * blockIdx.x;
    auto block_end = block_begin + block_len;
    if(block_end > total_len){
      block_end = total_len;
    }


    if(block_begin < total_len){

      // How many batches of threads do we need?
      ptrdiff_t num_batches = block_len / exec_info.threads_left;
      if(num_batches*exec_info.threads_left < block_len){
        num_batches++;
      }

      // compute our starting index
      ptrdiff_t i = exec_info.thread_id+block_begin;

      for(ptrdiff_t batch = 0;batch < num_batches;++ batch){

        if(i < block_end){
          // Compute indices from layout, and assign them to our index tuple
          layout.toIndices(i, make_index_assigner<Args>(data.segment_tuple, data.index_tuple)...);

          // invoke our enclosed statement list
          wrap(data, exec_info);
        }

        i += exec_info.threads_left;
      }

    }


  }
};



/*
 * Collapses multiple segments iteration space, and distributes them over
 * all of the blocks and threads.
 */
template <camp::idx_t ... Args, typename... EnclosedStmts>
struct CudaStatementExecutor<Collapse<cuda_block_seq_exec, ArgList<Args...>, EnclosedStmts...>> {

  static constexpr size_t num_dims = sizeof...(Args);

  template <typename WrappedBody, typename Data>
  static
  RAJA_DEVICE
  void exec(WrappedBody const &wrap, Data &data, CudaExecInfo &exec_info)
  {
    // Create a Layout of all of our loop dimensions that we're collapsing
    RAJA::Layout<num_dims> layout( (camp::get<Args>(data.segment_tuple).end() -
        camp::get<Args>(data.segment_tuple).begin()) ...);

    // get total iteration size
    ptrdiff_t total_len = layout.size();

    // compute our block's slice of work
    int num_blocks = gridDim.x;
    auto block_len = total_len / num_blocks;
    if(block_len*num_blocks < total_len){
      block_len ++;
    }
    auto block_begin = block_len * blockIdx.x;
    auto block_end = block_begin + block_len;
    if(block_end > total_len){
      block_end = total_len;
    }


    if(block_begin < total_len){

      // loop sequentially over our block
      for(ptrdiff_t i = block_begin;i < block_end;++ i){
        // Compute indices from layout, and assign them to our index tuple
        layout.toIndices(i, make_index_assigner<Args>(data.segment_tuple, data.index_tuple)...);

        // invoke our enclosed statement list
        wrap(data, exec_info);
      }

    }


  }
};



}  // namespace internal
}  // namespace nested
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
