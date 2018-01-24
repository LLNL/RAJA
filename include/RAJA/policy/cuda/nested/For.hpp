#ifndef RAJA_policy_cuda_nested_For_HPP
#define RAJA_policy_cuda_nested_For_HPP

#include "RAJA/config.hpp"
#include "RAJA/policy/cuda/nested.hpp"



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




namespace internal{





/*
 * Executor for sequential loops inside of a Cuda Kernel.
 */
template <camp::idx_t ArgumentId, typename... EnclosedStmts, typename IndexCalc>
struct CudaStatementExecutor<For<ArgumentId, seq_exec, EnclosedStmts...>, IndexCalc> {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  template <typename Data>
  static
  inline
  RAJA_DEVICE
  void exec(Data &data, long logical_block)
  {
    // Get the segment referenced by this For statement
    auto const &iter = camp::get<ArgumentId>(data.segment_tuple);

    // Pull out iterators
    auto begin = iter.begin();
    auto end = iter.end();

    // compute trip count
    auto len = end - begin;

    // sequentially step through indices
    // since we aren't assigning threads, we pass thru the IndexCalc, and
    // directly assign to the index_tuple
    for (decltype(len) i = 0; i < len; ++i) {

      // assign index
      data.template assign_index<ArgumentId>(*(begin+i));

      // execute enclosed statements
      cuda_execute_statement_list<stmt_list_t, IndexCalc>(data, logical_block);
    }
  }


  template<typename Data>
  static
  RAJA_INLINE
  LaunchDim calculateDimensions(Data const &data, LaunchDim const &max_physical){

    // Return launch dimensions of enclosed statements
    return cuda_calcdims_statement_list<stmt_list_t, IndexCalc>(data, max_physical);
  }


};



/*
 * Executor for thread work sharing loops inside of a Cuda Kernel.
 *
 * No block work-sharing is applied
 */
template <camp::idx_t ArgumentId, typename... EnclosedStmts, typename IndexCalc>
struct CudaStatementExecutor<For<ArgumentId, cuda_thread_exec, EnclosedStmts...>, IndexCalc> {

  using stmt_list_t = StatementList<EnclosedStmts...>;
  using index_calc_t = ExtendCudaIndexCalc<IndexCalc, ArgumentId, cuda_thread_exec>;

  template <typename Data>
  static
  inline
  __device__
  void exec(Data &data, long logical_block)
  {
    // execute enclosed statements
    cuda_execute_statement_list<stmt_list_t, index_calc_t>(data, logical_block);
  }


  template<typename Data>
  static
  RAJA_INLINE
  LaunchDim calculateDimensions(Data const &data, LaunchDim const &max_physical){

    // Return launch dimensions of enclosed statements
    return cuda_calcdims_statement_list<stmt_list_t, index_calc_t>(data, max_physical);
  }


};





/*
 * Executor for block work sharing loops inside of a Cuda Kernel.
 *
 * No thread work-sharing is applied
 */
template <camp::idx_t ArgumentId, typename... EnclosedStmts, typename IndexCalc>
struct CudaStatementExecutor<For<ArgumentId, cuda_block_exec, EnclosedStmts...>, IndexCalc> {

  using stmt_list_t = StatementList<EnclosedStmts...>;
  using index_calc_t = ExtendCudaIndexCalc<IndexCalc, ArgumentId, cuda_block_exec>;

  template <typename Data>
  static
  inline
  __device__
  void exec(Data &data, long logical_block)
  {
    // execute enclosed statements
    cuda_execute_statement_list<stmt_list_t, index_calc_t>(data, logical_block);
  }


  template<typename Data>
  static
  RAJA_INLINE
  LaunchDim calculateDimensions(Data const &data, LaunchDim const &max_physical){

    // Return launch dimensions of enclosed statements
    return cuda_calcdims_statement_list<stmt_list_t, index_calc_t>(data, max_physical);
  }

};




/*
 * Executor for thread and block work sharing loops inside of a Cuda Kernel.
 *
 */
template <camp::idx_t ArgumentId, size_t num_blocks, typename... EnclosedStmts, typename IndexCalc>
struct CudaStatementExecutor<For<ArgumentId, cuda_block_thread_exec<num_blocks>, EnclosedStmts...>, IndexCalc> {


  using stmt_list_t = StatementList<EnclosedStmts...>;
  using index_calc_t = ExtendCudaIndexCalc<IndexCalc, ArgumentId, cuda_block_thread_exec<num_blocks>>;

  template <typename Data>
  static
  inline
  __device__
  void exec(Data &data, long logical_block)
  {
    // execute enclosed statements
    cuda_execute_statement_list<stmt_list_t, index_calc_t>(data, logical_block);
  }



  template<typename Data>
  static
  RAJA_INLINE
  LaunchDim calculateDimensions(Data const &data, LaunchDim const &max_physical){

    // Return launch dimensions of enclosed statements
    return cuda_calcdims_statement_list<stmt_list_t, index_calc_t>(data, max_physical);
  }

};






///*
// * Executor for thread and block work sharing loops inside of a Cuda Kernel.
// *
// */
//template <camp::idx_t ArgumentId, typename... EnclosedStmts>
//struct CudaStatementExecutor<For<ArgumentId, cuda_block_seq_exec, EnclosedStmts...>> {
//
//  template<typename SegmentTuple>
//  RAJA_INLINE
//  RAJA_HOST_DEVICE
//  static long getLength(SegmentTuple const &segments){
//
//    // Get the segment referenced by this For statement
//    auto const &iter = camp::get<ArgumentId>(segments);
//
//    // Pull out iterators
//    auto begin = iter.begin();
//    auto end = iter.end();
//
//    // compute trip count
//    return end - begin;
//  }
//
//  template <typename WrappedBody, typename Data, typename IndexCalc>
//  static
//  RAJA_DEVICE
//  void exec(WrappedBody const &wrap, Data &data, IndexCalc const &index_calc)
//  {
//    // Get the segment referenced by this For statement
//    auto const &iter = camp::get<ArgumentId>(data.segment_tuple);
//
//    // Pull out iterators
//    auto begin = iter.begin();
//    auto end = iter.end();
//
//    // compute trip count
//    ptrdiff_t total_len = end - begin;
//
//    // compute our block's slice of work
//    int num_blocks = gridDim.x;
//    auto block_len = total_len / num_blocks;
//    if(block_len*num_blocks < total_len){
//      block_len ++;
//    }
//    auto block_begin = block_len * blockIdx.x;
//    auto block_end = block_begin + block_len;
//    if(block_end > total_len){
//      block_end = total_len;
//    }
//
//
//
//    if(block_begin < total_len){
//
//      // loop sequentially over our block
//      for(ptrdiff_t i = block_begin;i < block_end;++ i){
//        data.template assign_index<ArgumentId>(*(begin+i));
//        wrap(data, index_calc);
//      }
//
//    }
//  }
//
//
//  template<typename SegmentTuple>
//  RAJA_INLINE
//  static LaunchDim getRequested(SegmentTuple const &segments, long max_physical_blocks, LaunchDim const &used){
//
//    // Get the segment referenced by this For statement
//    auto const &iter = camp::get<ArgumentId>(segments);
//
//    // Pull out iterators
//    auto begin = iter.begin();
//    auto end = iter.end();
//
//    // compute trip count
//    long total_len = end - begin;
//
//    // compute dimensions we need
//    LaunchDim our_used = used * cuda_block_seq_exec::calcBlocksThreads(max_physical_blocks, used.blocks, total_len);
//
//    // recurse
//    return cuda_get_statement_list_requested<SegmentTuple, EnclosedStmts...>(segments, max_physical_blocks, our_used);
//  }
//};



} // namespace internal
}  // end namespace nested
}  // end namespace RAJA



#endif /* RAJA_pattern_nested_HPP */
