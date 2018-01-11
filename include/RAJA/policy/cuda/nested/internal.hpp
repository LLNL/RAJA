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



struct LaunchDim {

  long blocks;
  long threads;

  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr
  LaunchDim() : blocks(1), threads(1) {}


  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr
  LaunchDim(long b, long t) : blocks(b), threads(t){}


  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr
  LaunchDim(LaunchDim const &c) : blocks(c.blocks), threads(c.threads) {}



  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr
  LaunchDim maximum(LaunchDim const & c) const {
    return LaunchDim{
      blocks > c.blocks ? blocks : c.blocks,
      threads > c.threads ? threads : c.threads};
  }

  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr
  LaunchDim operator*(LaunchDim const & c) const {
    return LaunchDim{blocks*c.blocks, threads*c.threads};
  }

};

struct CudaLaunchLimits {
  LaunchDim max_dims;
  LaunchDim physical_dims;
};



struct CudaIndexCalc_Terminator {

  long num_blocks;
  long block;

  template<typename Data>
  inline
  __device__
  void calcIndex(Data &, long ) const {}


  constexpr
  inline
  __device__
  long numThreads() const {
    return 1;
  }

  constexpr
  inline
  __device__
  long numBlocks() const {
    return num_blocks;
  }

  constexpr
  inline
  __device__
  long getBlock() const {
    return block;
  }
};

template<camp::idx_t ArgumentId, typename Parent>
struct CudaIndexCalc_Simple {
  long num_threads;
  Parent const &parent;

  constexpr
  __device__
  CudaIndexCalc_Simple(long nthreads, Parent const &p) : num_threads(nthreads), parent{p} {}

  template<typename Data>
  inline
  __device__
  void calcIndex(Data &data, long remainder) const {
    // Compute and assign our index
    auto begin = camp::get<ArgumentId>(data.segment_tuple).begin();
    long i = remainder % num_threads;
    data.template assign_index<ArgumentId>(*(begin+i));

    // Pass on remainder to parent
    parent.calcIndex(data, remainder / num_threads);
  }


  constexpr
  __device__
  long numThreads() const {
    return num_threads * parent.numThreads();
  }

  constexpr
  inline
  __device__
  long numBlocks() const {
    return parent.numBlocks();
  }

  constexpr
  inline
  __device__
  long getBlock() const {
    return parent.getBlock();
  }

};


template<camp::idx_t ArgumentId, typename Parent>
struct CudaIndexCalc_Offset {
  long num_threads;
  long offset;
  Parent const &parent;

  constexpr
  __device__
  CudaIndexCalc_Offset(long nthreads, long off, Parent const &p) : num_threads(nthreads), offset(off), parent{p} {}

  template<typename Data>
  inline
  __device__
  void calcIndex(Data &data, long remainder) const {
    // Compute and assign our index
    auto begin = camp::get<ArgumentId>(data.segment_tuple).begin();
    long i = remainder % num_threads;
    data.template assign_index<ArgumentId>(*(begin+i+offset));

    // Pass on remainder to parent
    parent.calcIndex(data, remainder / num_threads);
  }


  constexpr
  __device__
  long numThreads() const {
    return num_threads * parent.numThreads();
  }


  constexpr
  inline
  __device__
  long numBlocks() const {
    return parent.numBlocks();
  }

  constexpr
  inline
  __device__
  long getBlock() const {
    return parent.getBlock();
  }

};



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




template<typename Args, typename Layout, typename Parent>
struct CudaIndexCalc_Layout;

template<camp::idx_t ... Args, typename Layout, typename Parent>
struct CudaIndexCalc_Layout<ArgList<Args...>, Layout, Parent> {

  static_assert(sizeof...(Args) == Layout::n_dims, "");

  Layout const &layout;
  long num_threads;
  long offset;
  Parent const &parent;

  constexpr
  __device__
  CudaIndexCalc_Layout(Layout const &l, Parent const &p) : layout(l), num_threads(l.size()), offset(0), parent{p} {}

  constexpr
  __device__
  CudaIndexCalc_Layout(Layout const &l, long nt, long off, Parent const &p) : layout(l), num_threads(nt), offset(off), parent{p} {}


  template<typename Data>
  inline
  __device__
  void calcIndex(Data &data, long remainder) const {

    // Compute and assign our index
    layout.toIndices((remainder%num_threads)+offset, make_index_assigner<Args>(data.segment_tuple, data.index_tuple)...);


    // Pass on remainder to parent
    parent.calcIndex(data, remainder / num_threads);
  }


  constexpr
  __device__
  long numThreads() const {
    return num_threads * parent.numThreads();
  }

};




template <typename Policy>
struct CudaStatementExecutor{};

template <camp::idx_t idx, camp::idx_t N, typename StmtList, typename IndexCalc>
struct CudaStatementListExecutor;



template<typename StmtList, typename Data, typename IndexCalc>
RAJA_DEVICE
RAJA_INLINE
void cuda_execute_statement_list(Data &data, IndexCalc const &index_calc){

  CudaStatementListExecutor<0, StmtList::size, StmtList, IndexCalc>::exec(data, index_calc);

}





template <typename StmtList, typename Data>
struct CudaStatementListWrapper {

  template<typename IndexCalc>
  inline
  __device__
  void operator()(Data &data, IndexCalc const &index_calc) const
  {
    cuda_execute_statement_list<StmtList>(data, index_calc);
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


template <camp::idx_t statement_index, camp::idx_t num_statements, typename StmtList, typename IndexCalc>
struct CudaStatementListExecutor{

  template<typename Data>
  static
  inline
  __device__
  void exec(Data &data, IndexCalc const &index_calc){

    // Get the statement we're going to execute
    using statement = camp::at_v<StmtList, statement_index>;

    // Create a wrapper for enclosed statements within statement
    using eclosed_statements_t = typename statement::enclosed_statements_t;
    auto enclosed_wrapper = cuda_make_statement_list_wrapper<eclosed_statements_t>(data);

    // Execute this statement
    CudaStatementExecutor<statement>::exec(enclosed_wrapper, data, index_calc);

    // call our next statement
    CudaStatementListExecutor<statement_index+1, num_statements, StmtList, IndexCalc>::exec(data, index_calc);
  }


};


/*
 * termination case, a NOP.
 */

template <camp::idx_t num_statements, typename ... Stmts, typename IndexCalc>
struct CudaStatementListExecutor<num_statements,num_statements, StatementList<Stmts...>, IndexCalc> {

  template<typename Data>
  static
  inline
  __device__
  void exec(Data &, IndexCalc const &) {}


  template<typename SegmentTuple>
  RAJA_INLINE
  static LaunchDim getRequested(SegmentTuple const &segments, long max_physical_blocks, LaunchDim const &used){
    LaunchDim dims;
    VarOps::ignore_args(
      (dims = dims.maximum(
          CudaStatementExecutor<Stmts>::getRequested(segments, max_physical_blocks, used))
      )...
    );
    return dims;
  }
};






template<typename SegmentTuple, typename ... EnclosedStmts>
RAJA_INLINE
LaunchDim cuda_get_statement_list_requested(SegmentTuple const &segments, long max_physical_blocks, LaunchDim const &used){

  using StmtList = StatementList<EnclosedStmts...>;
  using IndexCalc = CudaIndexCalc_Terminator;

  using StmtListExec = CudaStatementListExecutor<StmtList::size, StmtList::size, StmtList, IndexCalc>;

  return StmtListExec::getRequested(segments, max_physical_blocks, used);
}




}  // namespace internal
}  // namespace nested
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
