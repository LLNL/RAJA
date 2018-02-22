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
 * Helper class allows using RAJA::Layout to assign indices directly in to the
 * LoopData::index_tuple.
 *
 * It takes an iterator and reference to a value in the index_tuple, and upon
 * assignment it sets the index_tuple value to iter[v].
 */
template<typename Iter, typename Value>
struct IndexAssigner{
  using Self = IndexAssigner<Iter, Value>;

  Iter iter;
  Value &value;

  RAJA_HOST_DEVICE
  RAJA_INLINE
  constexpr
  IndexAssigner(Iter const &i, Value &v) : iter(i), value(v){}


  template<typename T>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  Self &operator=(T v){
    value = iter[v];
    return *this;
  }

};

/*
 * Helper function to make creating IndexAssigner objects.
 */
template<typename Iter, typename Value>
RAJA_INLINE
RAJA_HOST_DEVICE
auto make_index_assigner(Iter && i, Value && v) ->
  IndexAssigner<Iter, Value>
{
  return IndexAssigner<Iter, Value>(std::forward<Iter>(i), std::forward<Value>(v));
}



/*
 * Collapse policy base class.
 *
 * Provides functionality to create Layout objects, and perform the
 * linear-to-indices conversion and setting of the index_tuple
 */
template<camp::idx_t ... Args>
struct CudaIndexCalc_CollapsePolicyBase {
  static constexpr size_t num_dims = sizeof...(Args);
  using layout_t = RAJA::Layout<num_dims, int, num_dims-1>;
  layout_t layout;

  template<typename SegmentTuple>
  static
  RAJA_HOST_DEVICE
  RAJA_INLINE
  layout_t createLayout(SegmentTuple const &segments){
    return layout_t( (camp::get<Args>(segments).end() -
        camp::get<Args>(segments).begin()) ...);
  }

  template<typename SegmentTuple>
  RAJA_INLINE
  RAJA_HOST_DEVICE
  CudaIndexCalc_CollapsePolicyBase(SegmentTuple const &segments) :
    layout(createLayout(segments))
  {
  }

  RAJA_INLINE
  RAJA_HOST_DEVICE
  int size() const {
    return (int)layout.size();
  }


  template<typename Data>
  RAJA_INLINE
  RAJA_DEVICE
  void assignIndex(Data &data, int i){
    layout.toIndices(i,
          make_index_assigner(camp::get<Args>(data.segment_tuple).begin(), camp::get<Args>(data.index_tuple))...
      );
  }
};




template<typename Args, typename ExecPolicy>
struct CudaIndexCalc_CollapsePolicy;


/*!
 *  Collapsing policy index calculator that maps all indices to threads.
 */
template<camp::idx_t ... Args>
struct CudaIndexCalc_CollapsePolicy<ArgList<Args...>, cuda_thread_exec> :
public CudaIndexCalc_CollapsePolicyBase<Args...>
{

  using Base = CudaIndexCalc_CollapsePolicyBase<Args...>;

  int i;

  template<typename SegmentTuple>
  RAJA_INLINE
  RAJA_HOST_DEVICE
  CudaIndexCalc_CollapsePolicy(SegmentTuple const &segments, LaunchDim const &) :
    Base(segments), i(0)
  {
  }


  template<typename Data>
  RAJA_INLINE
  RAJA_DEVICE
  int assignBegin(Data &data, int carry){
    i = 0;
    return increment(data, carry);
  }

  template<typename Data>
  RAJA_INLINE
  RAJA_DEVICE
  int increment(Data &data, int carry_in){
    int len = Base::size();
    i += carry_in;

    int carry_out = i / len;
    i = i - carry_out*len;  // i % len

    // Compute and assign our loop indices
    Base::assignIndex(data, i);

    return carry_out;
  }


};


#if 0

/*!
 *  Collapsing policy index calculator that maps all indices to blocks.
 */
template<camp::idx_t ... Args>
struct CudaIndexCalc_CollapsePolicy<ArgList<Args...>, cuda_block_exec> :
public CudaIndexCalc_CollapsePolicyBase<Args...>
{

  using Base = CudaIndexCalc_CollapsePolicyBase<Args...>;


  template<typename SegmentTuple>
  RAJA_INLINE
  RAJA_HOST_DEVICE
  CudaIndexCalc_CollapsePolicy(SegmentTuple const &segments, LaunchDim const &) :
    Base(segments)
  {
  }


  template<typename Data>
  RAJA_INLINE
  RAJA_DEVICE
  bool assignIndex(Data &data, int *block, int *){

    // Compute our linear index, and strip off the thread index
    int len = Base::layout.size();
    int i = (*block) % len;
    (*block) /= len;

    // Compute and assign our loop indices
    Base::assignIndex(data, i);

    return true;
  }
};
#endif




#if 0
/*!
 *  Collapsing policy index calculator that maps all indices to threads and blocks.
 */
template<camp::idx_t ... Args, size_t num_blocks_max>
struct CudaIndexCalc_CollapsePolicy<ArgList<Args...>, cuda_threadblock_exec<num_blocks_max>> :
public CudaIndexCalc_CollapsePolicyBase<Args...>
{

  using Base = CudaIndexCalc_CollapsePolicyBase<Args...>;

  int num_blocks;
  int num_threads;


  template<typename SegmentTuple>
  RAJA_INLINE
  RAJA_HOST_DEVICE
  CudaIndexCalc_CollapsePolicy(SegmentTuple const &segments, LaunchDim const &) :
    Base(segments),
    num_blocks(num_blocks_max < Base::layout.size() ? num_blocks_max : Base::layout.size()),
    num_threads(Base::layout.size()/num_blocks)
  {
    if(num_threads*num_blocks < Base::layout.size()){
      num_threads ++;
    }
  }


  RAJA_INLINE
  RAJA_HOST_DEVICE
  int numLogicalBlocks() const {
    return num_blocks;
  }

  RAJA_INLINE
  RAJA_HOST_DEVICE
  int numLogicalThreads() const {
    return num_threads;
  }


  template<typename Data>
  RAJA_INLINE
  RAJA_DEVICE
  bool assignIndex(Data &data, int *block, int *thread){

    // Compute our linear index, and strip off the block and thread indices
    int block_i = (*block) % num_blocks;
    (*block) /= num_blocks;

    int thread_i = (*thread) % num_threads;
    (*thread) /= num_threads;

    int i = block_i*num_threads + thread_i;


    // Compute and assign our loop indices
    int len = Base::layout.size();
    if(i < len){
      Base::assignIndex(data, i);
      return true;
    }

    return false;
  }
};


#endif



/*
 * Statement Executor for collapsing multiple segments iteration space,
 * and provides work sharing according to the collapsing execution policy.
 */
template <camp::idx_t ... Args, typename... EnclosedStmts, typename IndexCalc>
struct CudaStatementExecutor<Collapse<cuda_thread_exec, ArgList<Args...>, EnclosedStmts...>, IndexCalc> {

  using stmt_list_t = StatementList<EnclosedStmts...>;
  using collapse_policy_t = CudaIndexCalc_CollapsePolicy<ArgList<Args...>, cuda_thread_exec>;
  using index_calc_t = ExtendCudaIndexCalc<IndexCalc, collapse_policy_t>;


  template <typename Data>
  static
  inline
  __device__
  void exec(Data &data, int num_logical_blocks, int logical_block)
  {
    // execute enclosed statements
    cuda_execute_statement_list<stmt_list_t, index_calc_t>(data, num_logical_blocks, logical_block);

  }


  template<typename Data>
  static
  RAJA_INLINE
  LaunchDim calculateDimensions(Data const &data, LaunchDim const &max_physical){

    // Get launch dimensions of enclosed statements
    LaunchDim dim = cuda_calcdims_statement_list<stmt_list_t, IndexCalc>(data, max_physical);

    // Append the number of threads we generate
    collapse_policy_t cpol(data.segment_tuple, max_physical);
    dim.threads *= cpol.size();

    return dim;
  }

};







}  // namespace internal
}  // namespace nested
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
