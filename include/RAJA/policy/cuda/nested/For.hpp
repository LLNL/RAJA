#ifndef RAJA_policy_cuda_nested_For_HPP
#define RAJA_policy_cuda_nested_For_HPP

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


namespace internal{



/*
 * Executor for thread work sharing loop inside a Cuda Kernel.
 *
 */
template <camp::idx_t ArgumentId, typename... EnclosedStmts, typename IndexCalc>
struct CudaStatementExecutor<For<ArgumentId, cuda_thread_exec, EnclosedStmts...>, IndexCalc> {

  using stmt_list_t = StatementList<EnclosedStmts...>;
  using index_calc_t = ExtendCudaIndexCalc<IndexCalc,CudaIndexCalc_Policy<ArgumentId, cuda_thread_exec>>;

  template <typename Data>
  static
  inline
  RAJA_DEVICE
  void exec(Data &data, int num_logical_blocks, int logical_block)
  {
    // execute enclosed statements
		cuda_execute_statement_list<stmt_list_t, index_calc_t>(data, num_logical_blocks, logical_block);
  }

  template<typename Data>
  static
  RAJA_INLINE
	LaunchDim calculateDimensions(Data const &data, LaunchDim const &max_physical){
		
		LaunchDim dim = cuda_calcdims_statement_list<stmt_list_t, IndexCalc>(data, max_physical);
		
		dim.threads *= segment_length<ArgumentId>(data);
		
		return dim;
	}
};

/*
 * Executor for block work sharing loop inside a Cuda Kernel.
 *
 */
template <camp::idx_t ArgumentId, typename... EnclosedStmts, typename IndexCalc>
struct CudaStatementExecutor<For<ArgumentId, cuda_block_exec, EnclosedStmts...>, IndexCalc> {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  template <typename Data>
  static
  inline
  RAJA_DEVICE
  void exec(Data &data, int num_logical_blocks, int logical_block)
  {
	
		// Distribute work over blocks using 1 thread per block
		cuda_execute_block_loop<ArgumentId, stmt_list_t, IndexCalc, 1>(data, num_logical_blocks, logical_block);

  }

  template<typename Data>
  static
  RAJA_INLINE
	LaunchDim calculateDimensions(Data const &data, LaunchDim const &max_physical){
		
		LaunchDim dim = cuda_calcdims_statement_list<stmt_list_t, IndexCalc>(data, max_physical);
		
		dim.blocks *= segment_length<ArgumentId>(data);
		
		return dim;
	}
};


/*
 * Executor for thread and block work sharing loop inside a Cuda Kernel.
 *
 */
template <camp::idx_t ArgumentId, size_t max_threads, typename... EnclosedStmts, typename IndexCalc>
struct CudaStatementExecutor<For<ArgumentId, cuda_threadblock_exec<max_threads>, EnclosedStmts...>, IndexCalc> {

  using stmt_list_t = StatementList<EnclosedStmts...>;
  using index_calc_t = ExtendCudaIndexCalc<IndexCalc,CudaIndexCalc_Policy<ArgumentId, cuda_thread_exec>>;

  
	template <typename Data>
  static
  inline
  RAJA_DEVICE
  void exec(Data &data, int num_logical_blocks, int logical_block)
  {
		// Distribute work over blocks using max_threads thread per block
		cuda_execute_block_loop<ArgumentId, stmt_list_t, index_calc_t, max_threads>(data, num_logical_blocks, logical_block);
		
  }

  template<typename Data>
  static
  RAJA_INLINE
	LaunchDim calculateDimensions(Data const &data, LaunchDim const &max_physical){
		
		LaunchDim dim = cuda_calcdims_statement_list<stmt_list_t, IndexCalc>(data, max_physical);
		
		// Compute how many blocks
		int len = segment_length<ArgumentId>(data);
		int num_blocks = len / max_threads;
		if(num_blocks*max_threads < len){
			num_blocks ++;
		}
		
		dim.blocks *= num_blocks;
		dim.threads *= (int)max_threads;
		
		return dim;
	}
};





/*
 * Executor for sequential loops inside of a Cuda Kernel.
 *
 * This is specialized since it need to execute the loop immediately.
 */
template <camp::idx_t ArgumentId, typename... EnclosedStmts, typename IndexCalc>
struct CudaStatementExecutor<For<ArgumentId, seq_exec, EnclosedStmts...>, IndexCalc> {

  using stmt_list_t = StatementList<EnclosedStmts...>;
  using index_calc_t = ExtendCudaIndexCalc<IndexCalc,CudaIndexCalc_Policy<ArgumentId, seq_exec>>;

  template <typename Data>
  static
  inline
	RAJA_DEVICE
  void exec(Data &data, int num_logical_blocks, int logical_block)
  {
		cuda_execute_statement_list<stmt_list_t, index_calc_t>(data, num_logical_blocks, logical_block);
  }

  template<typename Data>
  static
  RAJA_INLINE
  LaunchDim calculateDimensions(Data const &data, LaunchDim const &max_physical){

    // Return launch dimensions of enclosed statements
		// seq_exec doesn't affect the number of threads or blocks
    return cuda_calcdims_statement_list<stmt_list_t, IndexCalc>(data, max_physical);
  }
};


template <camp::idx_t ArgumentId, typename... EnclosedStmts, typename Segments>
struct CudaStatementExecutor<For<ArgumentId, seq_exec, EnclosedStmts...>, CudaIndexCalc_Terminator<Segments>> {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  template <typename Data>
  static
  inline
  RAJA_DEVICE
  void exec(Data &data, int num_logical_blocks, int logical_block)
  {
    int len = segment_length<ArgumentId>(data);
    auto begin = camp::get<ArgumentId>(data.segment_tuple).begin();

    for(int i = 0;i < len;++ i){
      data.template assign_index<ArgumentId>(*(begin+i));
      cuda_execute_statement_list<stmt_list_t, CudaIndexCalc_Terminator<Segments>>(data, num_logical_blocks, logical_block);
    }
  }

  template<typename Data>
  static
  RAJA_INLINE
  LaunchDim calculateDimensions(Data const &data, LaunchDim const &max_physical){

    // Return launch dimensions of enclosed statements
    // seq_exec doesn't affect the number of threads or blocks
    return cuda_calcdims_statement_list<stmt_list_t, CudaIndexCalc_Terminator<Segments>>(data, max_physical);
  }
};




} // namespace internal
}  // end namespace nested
}  // end namespace RAJA



#endif /* RAJA_pattern_nested_HPP */
