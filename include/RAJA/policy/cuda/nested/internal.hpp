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


/*!
 * Policy for For<>, executes loop iteration by distributing them over threads.
 * This does no (additional) work-sharing between thread blocks.
 */

struct cuda_thread_exec{};




/*!
 * Policy for For<>, executes loop iteration by distributing iterations
 * exclusively over blocks.
 */

struct cuda_block_exec{};




/*!
 * Policy for For<>, executes loop iteration by distributing them over all
 * blocks and threads.
 */

template<size_t num_blocks>
struct cuda_block_seq_exec{};





/*!
 * Policy for For<>, executes loop iteration by distributing them over all
 * blocks and then executing sequentially on each thread.
 */
template<size_t num_blocks>
struct cuda_block_thread_exec{};


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





template<camp::idx_t ArgumentId, typename ExecPolicy, typename SegmentType>
struct CudaIndexCalc_Policy;


template<camp::idx_t ArgumentId, typename SegmentType>
struct CudaIndexCalc_Policy<ArgumentId, cuda_block_exec, SegmentType>{

  using iterator_t = typename SegmentType::iterator;
  using difference_t = typename iterator_t::difference_type;

  iterator_t begin;
  difference_t len;

  RAJA_INLINE
  RAJA_HOST_DEVICE
  CudaIndexCalc_Policy(SegmentType const &s, LaunchDim const &) :
  begin(s.begin()), len(s.end()-s.begin())
  {
  }

  RAJA_INLINE
  RAJA_HOST_DEVICE
  LaunchDim cudaCalcDims(){
    return LaunchDim{(long)len, 1};
  }


  template<typename Data>
  RAJA_INLINE
  RAJA_DEVICE
  bool cudaAssignIndex(Data &data, LaunchDim &block_thread){

    // Compute our index, and strip off the block index
    long i = block_thread.blocks % len;
    block_thread.blocks /= len;

    // Assign our computed index to the tuple
    data.template assign_index<ArgumentId>(*(begin+i));

    return true;
  }

};





template<camp::idx_t ArgumentId, typename SegmentType>
struct CudaIndexCalc_Policy<ArgumentId, cuda_thread_exec, SegmentType>{

  using iterator_t = typename SegmentType::iterator;
  using difference_t = typename iterator_t::difference_type;

  iterator_t begin;
  difference_t len;

  RAJA_INLINE
  RAJA_HOST_DEVICE
  CudaIndexCalc_Policy(SegmentType const &s, LaunchDim const &) :
  begin(s.begin()), len(s.end()-s.begin())
  {
  }

  RAJA_INLINE
  RAJA_HOST_DEVICE
  LaunchDim cudaCalcDims(){
    return LaunchDim{1, (long)len};
  }


  template<typename Data>
  RAJA_INLINE
  RAJA_DEVICE
  bool cudaAssignIndex(Data &data, LaunchDim &block_thread){

    // Compute our index, and strip off the block index
    long i = block_thread.threads % len;
    block_thread.threads /= len;

    // Assign our computed index to the tuple
    data.template assign_index<ArgumentId>(*(begin+i));

    return true;
  }

};



template<camp::idx_t ArgumentId, size_t num_blocks_max, typename SegmentType>
struct CudaIndexCalc_Policy<ArgumentId, cuda_block_thread_exec<num_blocks_max>, SegmentType>{

  using iterator_t = typename SegmentType::iterator;
  using difference_t = typename iterator_t::difference_type;

  iterator_t begin;
  difference_t len;
  long num_blocks;
  long num_threads;

  RAJA_INLINE
  RAJA_HOST_DEVICE
  CudaIndexCalc_Policy(SegmentType const &s, LaunchDim const &) :
    begin(s.begin()),
    len(s.end()-s.begin()),
    num_blocks(num_blocks_max < len ? num_blocks_max : len),
    num_threads(len/num_blocks)
  {
    if(num_threads*num_blocks < len){
      num_threads ++;
    }
  }

  RAJA_INLINE
  RAJA_HOST_DEVICE
  LaunchDim cudaCalcDims(){
    return LaunchDim{num_blocks, num_threads};
  }


  template<typename Data>
  RAJA_INLINE
  RAJA_DEVICE
  bool cudaAssignIndex(Data &data, LaunchDim &block_thread){

    // Compute our index, and strip off the thread index
    long block_i = block_thread.blocks % num_blocks;
    block_thread.blocks /= num_blocks;

    long thread_i = block_thread.threads % num_threads;
    block_thread.threads /= num_threads;

    long i = block_i*num_threads + thread_i;

    // Assign our computed index to the tuple
    if(i < len){
      data.template assign_index<ArgumentId>(*(begin+i));
      return true;
    }
    // our i is out of bounds
    return false;
  }

};





template<typename SegmentTuple, typename ArgList, typename ExecPolicies, typename RangeList>
struct CudaIndexCalc;


template<typename SegmentTuple, camp::idx_t ... Args, typename ... ExecPolicies, camp::idx_t ... RangeInts>
struct CudaIndexCalc<SegmentTuple, ArgList<Args...>, camp::list<ExecPolicies...>, camp::idx_seq<RangeInts...>>{

  using CalcList = camp::tuple<CudaIndexCalc_Policy<Args, ExecPolicies, camp::at_v<typename SegmentTuple::TList, Args>>...>;

  CalcList calc_list;

  RAJA_INLINE
  RAJA_HOST_DEVICE
  CudaIndexCalc(SegmentTuple const &segment_tuple, LaunchDim const &max_physical) :
    calc_list( camp::make_tuple( (camp::at_v<typename CalcList::TList, RangeInts>(camp::get<Args>(segment_tuple), max_physical) )...) )
  {
  }

  RAJA_INLINE
  RAJA_HOST_DEVICE
  LaunchDim computeLogicalDims(){

    // evaluate product of each Calculator's block and thread requirements
    return VarOps::foldl(RAJA::operators::multiplies<LaunchDim>(),
        camp::get<RangeInts>(calc_list).cudaCalcDims()...);

  }

  template<typename Data>
  RAJA_INLINE
  RAJA_DEVICE
  bool assignIndices(Data &data, LaunchDim block_thread){

    // evaluate each index, passing block_thread through
    // each calculator will trim block_thread appropriately
    return VarOps::foldl(RAJA::operators::logical_and<bool>(),
        camp::get<RangeInts>(calc_list).cudaAssignIndex(data, block_thread)...);
  }

};

template<typename SegmentTuple>
using CudaIndexCalc_Terminator = CudaIndexCalc<SegmentTuple, ArgList<>, camp::list<>, camp::idx_seq<>>;


template<typename IndexCalcBase, camp::idx_t ArgN, typename CalcN>
struct CudaIndexCalc_Extender;


template<typename SegmentTuple, camp::idx_t ... Args, typename ... CalcTypes, camp::idx_t ... RangeInts, camp::idx_t ArgN, typename CalcN>
struct CudaIndexCalc_Extender<CudaIndexCalc<SegmentTuple, ArgList<Args...>, camp::list<CalcTypes...>, camp::idx_seq<RangeInts...>>, ArgN, CalcN>{
  using type = CudaIndexCalc<SegmentTuple, ArgList<Args..., ArgN>, camp::list<CalcTypes..., CalcN>, camp::idx_seq<RangeInts..., sizeof...(RangeInts)>>;
};

template<typename IndexCalcBase, camp::idx_t ArgN, typename CalcN>
using ExtendCudaIndexCalc = typename CudaIndexCalc_Extender<IndexCalcBase, ArgN, CalcN>::type;








template <typename Policy, typename IndexCalc>
struct CudaStatementExecutor;

template <camp::idx_t idx, camp::idx_t N, typename StmtList, typename IndexCalc>
struct CudaStatementListExecutor;


template <camp::idx_t statement_index, camp::idx_t num_statements, typename StmtList, typename IndexCalc>
struct CudaStatementListExecutor{

  template<typename Data>
  static
  inline
  __device__
  void exec(Data &data, long logical_block){

    // Get the statement we're going to execute
    using statement = camp::at_v<StmtList, statement_index>;

    // Execute this statement
    CudaStatementExecutor<statement, IndexCalc>::exec(data, logical_block);

    // call the next statement in the list
    CudaStatementListExecutor<statement_index+1, num_statements, StmtList, IndexCalc>::exec(data, logical_block);
  }


  template<typename Data>
  static
  RAJA_INLINE
  LaunchDim calculateDimensions(Data const &data, LaunchDim const &max_physical){

    // Compute this statements launch dimensions
    using statement = camp::at_v<StmtList, statement_index>;
    LaunchDim statement_dims = CudaStatementExecutor<statement, IndexCalc>::calculateDimensions(data, max_physical);

    // call the next statement in the list
    LaunchDim next_dims = CudaStatementListExecutor<statement_index+1, num_statements, StmtList, IndexCalc>::calculateDimensions(data, max_physical);

    // Return the maximum of the two
    return statement_dims.maximum(next_dims);
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
  void exec(Data &, long) {}


  template<typename Data>
  static
  RAJA_INLINE
  LaunchDim calculateDimensions(Data const &, LaunchDim const &){

    return LaunchDim();
  }
};



template<typename StmtList, typename IndexCalc, typename Data>
RAJA_DEVICE
RAJA_INLINE
void cuda_execute_statement_list(Data &data, long logical_block){

  CudaStatementListExecutor<0, StmtList::size, StmtList, IndexCalc>::exec(data, logical_block);

}




template<typename StmtList, typename IndexCalc, typename Data>
RAJA_INLINE
LaunchDim cuda_calcdims_statement_list(Data const &data, LaunchDim const &max_physical){

  using StmtListExec = CudaStatementListExecutor<0, StmtList::size, StmtList, IndexCalc>;

  return StmtListExec::calculateDimensions(data, max_physical);
}


template<typename Data, typename ... EnclosedStmts>
RAJA_INLINE
LaunchDim cuda_calculate_logical_dimensions(Data const &data, LaunchDim const &max_physical){

  using index_calc_t = CudaIndexCalc_Terminator<typename Data::segment_tuple_t>;
  using stmt_list_t = StatementList<EnclosedStmts...>;

  return cuda_calcdims_statement_list<stmt_list_t, index_calc_t>(data, max_physical);

}



}  // namespace internal
}  // namespace nested
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
