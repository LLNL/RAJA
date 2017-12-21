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

/*!
 * Policy for For<>, executes loop iteration by distributing them over threads.
 * This does no (additional) work-sharing between thread blocks.
 */
struct cuda_thread_exec {};


/*!
 * Policy for For<>, executes loop iteration by distributing them over all
 * blocks and threads.
 */
struct cuda_block_thread_exec {};


/*!
 * Policy for For<>, executes loop iteration by distributing them over all
 * blocks and then executing sequentially on each thread.
 */
struct cuda_block_seq_exec {};

namespace nested
{




namespace internal{




/*
 * Executor for sequential loops inside of a Cuda Kernel.
 */
template <camp::idx_t ArgumentId, typename... EnclosedStmts>
struct CudaStatementExecutor<For<ArgumentId, seq_exec, EnclosedStmts...>> {

  using ForType = For<ArgumentId, seq_exec, EnclosedStmts...>;

  template <typename WrappedBody>
  RAJA_INLINE
  RAJA_DEVICE
  void operator()(ForType const &fp, WrappedBody const &wrap, CudaExecInfo &exec_info)
  {
    // Get the segment referenced by this For statement
    auto const &iter = camp::get<ArgumentId>(wrap.data.segment_tuple);

    // Pull out iterators
    auto begin = iter.begin();  // std::begin(iter);
    auto end = iter.end();      // std::end(iter);

    // compute trip count
    auto len = end - begin;  // std::distance(begin, end);


    for (decltype(len) i = 0; i < len; ++i) {
      wrap.data.template assign_index<ArgumentId>(i);
      wrap(exec_info);
    }
  }
};





/*
 * Executor for thread work sharing loops inside of a Cuda Kernel.
 *
 * No block work-sharing is applied
 */
template <camp::idx_t ArgumentId, typename... EnclosedStmts>
struct CudaStatementExecutor<For<ArgumentId, cuda_thread_exec, EnclosedStmts...>> {

  using ForType = For<ArgumentId, cuda_thread_exec, EnclosedStmts...>;

  template <typename WrappedBody>
  RAJA_INLINE
  RAJA_DEVICE
  void operator()(ForType const &fp, WrappedBody const &wrap, CudaExecInfo &exec_info)
  {
    // Get the segment referenced by this For statement
    auto const &iter = camp::get<ArgumentId>(wrap.data.segment_tuple);

    // Pull out iterators
    auto begin = iter.begin();  // std::begin(iter);
    auto end = iter.end();      // std::end(iter);

    // compute trip count
    auto len = end - begin;  // std::distance(begin, end);

    // How many batches of threads do we need?
    int num_batches = len / exec_info.threads_left;
    if(num_batches*exec_info.threads_left < len){
      num_batches++;
    }

    // compute our starting index
    int i = exec_info.thread_id;

    for(int batch = 0;batch < num_batches;++ batch){

      if(i < len){
        wrap.data.template assign_index<ArgumentId>(*(begin+i));
        wrap(exec_info);
      }

      i += exec_info.threads_left;
    }

  }
};




/*
 * Executor for thread and block work sharing loops inside of a Cuda Kernel.
 *
 */
template <camp::idx_t ArgumentId, typename... EnclosedStmts>
struct CudaStatementExecutor<For<ArgumentId, cuda_block_thread_exec, EnclosedStmts...>> {

  using ForType = For<ArgumentId, cuda_block_thread_exec, EnclosedStmts...>;

  template <typename WrappedBody>
  RAJA_INLINE
  RAJA_DEVICE
  void operator()(ForType const &fp, WrappedBody const &wrap, CudaExecInfo &exec_info)
  {
    // Get the segment referenced by this For statement
    auto const &iter = camp::get<ArgumentId>(wrap.data.segment_tuple);

    // Pull out iterators
    auto begin = iter.begin();  // std::begin(iter);
    auto end = iter.end();      // std::end(iter);

    // compute trip count
    ptrdiff_t total_len = end - begin;  // std::distance(begin, end);

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

//      if(threadIdx.x == 0){
//            printf("block_len=%ld, block_begin=%ld, block_end=%ld, num_batches=%ld, num_threads=%ld\n",
//                (long)block_len, (long)block_begin, (long)block_end,
//                (long)num_batches, (long)exec_info.threads_left);
//          }

      // compute our starting index
      ptrdiff_t i = exec_info.thread_id+block_begin;

      for(ptrdiff_t batch = 0;batch < num_batches;++ batch){

        if(i < block_end){
          wrap.data.template assign_index<ArgumentId>(*(begin+i));
          wrap(exec_info);
        }

        i += exec_info.threads_left;
      }

    }
  }
};





/*
 * Executor for thread and block work sharing loops inside of a Cuda Kernel.
 *
 */
template <camp::idx_t ArgumentId, typename... EnclosedStmts>
struct CudaStatementExecutor<For<ArgumentId, cuda_block_seq_exec, EnclosedStmts...>> {

  using ForType = For<ArgumentId, cuda_block_seq_exec, EnclosedStmts...>;

  template <typename WrappedBody>
  RAJA_INLINE
  RAJA_DEVICE
  void operator()(ForType const &fp, WrappedBody const &wrap, CudaExecInfo &exec_info)
  {
    // Get the segment referenced by this For statement
    auto const &iter = camp::get<ArgumentId>(wrap.data.segment_tuple);

    // Pull out iterators
    auto begin = iter.begin();  // std::begin(iter);
    auto end = iter.end();      // std::end(iter);

    // compute trip count
    ptrdiff_t total_len = end - begin;  // std::distance(begin, end);

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
        wrap.data.template assign_index<ArgumentId>(*(begin+i));
        wrap(exec_info);
      }

    }
  }
};


} // namespace internal
}  // end namespace nested
}  // end namespace RAJA



#endif /* RAJA_pattern_nested_HPP */
