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
 * Collapses multiple segments iteration space, and distributes them over threads.
 *
 * No work sharing between blocks is performed
 */
template <camp::idx_t ... Args, typename... EnclosedStmts>
struct CudaStatementExecutor<Collapse<cuda_thread_exec, ArgList<Args...>, EnclosedStmts...>> {

  static constexpr size_t num_dims = sizeof...(Args);


  template<typename SegmentTuple>
  static
  RAJA_HOST_DEVICE
  RAJA_INLINE
  RAJA::Layout<num_dims> getLayout(SegmentTuple const &segments){
    return RAJA::Layout<num_dims>( (camp::get<Args>(segments).end() -
        camp::get<Args>(segments).begin()) ...);
  }

  template <typename WrappedBody, typename Data, typename IndexCalc>
  static
  RAJA_DEVICE
  void exec(WrappedBody const &wrap, Data &data, IndexCalc const &parent_index_calc)
  {
    // Create a Layout of all of our loop dimensions that we're collapsing
    auto layout = getLayout(data.segment_tuple);

    CudaIndexCalc_ThreadLayout<ArgList<Args...>, RAJA::Layout<num_dims>, IndexCalc>
      index_calc(layout, parent_index_calc);


    //invoke our enclosed statement list
    wrap(data, index_calc);
  }


  template<typename SegmentTuple>
  RAJA_INLINE
  static LaunchDim getRequested(SegmentTuple const &segments, long max_physical_blocks, LaunchDim const &used){

    // compute trip count
    auto layout = getLayout(segments);
    auto total_len = layout.size();

    // compute dimensions we need
    LaunchDim our_used = used * cuda_thread_exec::calcBlocksThreads(max_physical_blocks, used.blocks, total_len);

    // recurse
    return cuda_get_statement_list_requested<SegmentTuple, EnclosedStmts...>(segments, max_physical_blocks, our_used);
  }

};




/*
 * Collapses multiple segments iteration space, and distributes them over
 * all of the blocks and threads.
 */
template <camp::idx_t ... Args, typename... EnclosedStmts>
struct CudaStatementExecutor<Collapse<cuda_block_thread_exec, ArgList<Args...>, EnclosedStmts...>> {

  static constexpr size_t num_dims = sizeof...(Args);

  template<typename SegmentTuple>
  static
  RAJA_HOST_DEVICE
  RAJA_INLINE
  RAJA::Layout<num_dims> getLayout(SegmentTuple const &segments){
    return RAJA::Layout<num_dims>( (camp::get<Args>(segments).end() -
        camp::get<Args>(segments).begin()) ...);
  }


  template <typename WrappedBody, typename Data, typename IndexCalc>
  static
  RAJA_DEVICE
  void exec(WrappedBody const &wrap, Data &data, IndexCalc const &parent_index_calc)
  {
    // Create a Layout of all of our loop dimensions that we're collapsing
    auto layout = getLayout(data.segment_tuple);

    // get total iteration size
    ptrdiff_t total_len = layout.size();

    // compute our block's slice of work
    int num_blocks = parent_index_calc.numLogicalBlocks();
    auto block_len = total_len / num_blocks;
    if(block_len*num_blocks < total_len){
      block_len ++;
    }
    auto block_begin = block_len * parent_index_calc.getLogicalBlock();
    auto block_end = block_begin + block_len;
    if(block_end > total_len){
      block_end = total_len;
    }


    // Create a Layout of all of our loop dimensions that we're collapsing
    CudaIndexCalc_ThreadLayout<ArgList<Args...>, RAJA::Layout<num_dims>, IndexCalc>
      index_calc(layout, block_end-block_begin, block_begin,  parent_index_calc);


    //invoke our enclosed statement list
    wrap(data, index_calc);
  }


  template<typename SegmentTuple>
  RAJA_INLINE
  static LaunchDim getRequested(SegmentTuple const &segments, long max_physical_blocks, LaunchDim const &used){

    // compute trip count
    auto layout = getLayout(segments);
    auto total_len = layout.size();

    // compute dimensions we need
    LaunchDim our_used = used * cuda_block_thread_exec::calcBlocksThreads(max_physical_blocks, used.blocks, total_len);

    // recurse
    return cuda_get_statement_list_requested<SegmentTuple, EnclosedStmts...>(segments, max_physical_blocks, our_used);
  }
};



/*
 * Collapses multiple segments iteration space, and distributes them over
 * all of the blocks and threads.
 */
template <camp::idx_t ... Args, typename... EnclosedStmts>
struct CudaStatementExecutor<Collapse<cuda_block_seq_exec, ArgList<Args...>, EnclosedStmts...>> {

  static constexpr size_t num_dims = sizeof...(Args);

  template<typename SegmentTuple>
  static
  RAJA_HOST_DEVICE
  RAJA_INLINE
  RAJA::Layout<num_dims> getLayout(SegmentTuple const &segments){
    return RAJA::Layout<num_dims>( (camp::get<Args>(segments).end() -
        camp::get<Args>(segments).begin()) ...);
  }

  template <typename WrappedBody, typename Data, typename IndexCalc>
  static
  RAJA_DEVICE
  void exec(WrappedBody const &wrap, Data &data, IndexCalc const &index_calc)
  {
    // Create a Layout of all of our loop dimensions that we're collapsing
    auto layout = getLayout(data.segment_tuple);

    // get total iteration size
    ptrdiff_t total_len = layout.size();

    // compute our block's slice of work
    long num_blocks = index_calc.numLogicalBlocks();
    auto block_len = total_len / num_blocks;
    if(block_len*num_blocks < total_len){
      block_len ++;
    }
    auto block_begin = block_len * index_calc.getLogicalBlock();
    auto block_end = block_begin + block_len;
    if(block_end > total_len){
      block_end = total_len;
    }

    // loop sequentially over our block
    for(ptrdiff_t i = block_begin;i < block_end;++ i){
      // Compute indices from layout, and assign them to our index tuple
      layout.toIndices(i, make_index_assigner<Args>(data.segment_tuple, data.index_tuple)...);

      // invoke our enclosed statement list
      wrap(data, index_calc);
    }
  }



  template<typename SegmentTuple>
  RAJA_INLINE
  static LaunchDim getRequested(SegmentTuple const &segments, long max_physical_blocks, LaunchDim const &used){

    // compute trip count
    auto layout = getLayout(segments);
    auto total_len = layout.size();

    // compute dimensions we need
    LaunchDim our_used = used * cuda_block_seq_exec::calcBlocksThreads(max_physical_blocks, used.blocks, total_len);

    // recurse
    return cuda_get_statement_list_requested<SegmentTuple, EnclosedStmts...>(segments, max_physical_blocks, our_used);
  }
};



}  // namespace internal
}  // namespace nested
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
