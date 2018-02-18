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
namespace internal
{

template <camp::idx_t LoopIndex, typename IndexCalc>
struct CudaStatementExecutor<Lambda<LoopIndex>, IndexCalc>{

  template <typename Data>
  static
  //__noinline__
  inline
  __device__
  void exec(Data &data, int num_logical_blocks, int logical_block)
  
	{
//		printf("Lambda<%d> nlb=%d, lb=%d\n", (int)LoopIndex,  num_logical_blocks, logical_block);
		if(logical_block <= 0){
			// Get physical parameters
			LaunchDim max_physical(gridDim.x, blockDim.x);

			// Compute logical dimensions
			IndexCalc index_calc(data.segment_tuple, max_physical);
			
			// set indices to beginning of each segment, and increment
			// to this threads first iteration
			bool done = index_calc.assignBegin(data, threadIdx.x);
			
			
			

			// Loop over logical threads in this block
			int iter = 0;
			while(!done) {
			/*
			if(threadIdx.x>1){
				extern __shared__ long win[];
				int z = *camp::get<3>(data.index_tuple) - win[3];
				assert(z >= 0);
				assert(z < 16);
				int m = *camp::get<1>(data.index_tuple) - win[1];
				assert(m >= 0);
				assert(m < 25);
				int d = *camp::get<2>(data.index_tuple) - win[2];
				assert(d >= 0);
				assert(d < 80);
			}*/

			/*
				extern __shared__ long win[];
				int x = *camp::get<3>(data.index_tuple) - win[3];
				assert(x >= 0);
				assert(x < 16);
*/
//			__threadfence();
//			__syncthreads();
				//printf("B/T=%d,%d iter=%d\n", (int)blockIdx.x, (int)threadIdx.x, iter);
				invoke_lambda<LoopIndex>(data);
				
				++ iter;
				done = index_calc.increment(data, blockDim.x);
			}
			__threadfence();
			__syncthreads();
//			if(blockIdx.x==0)
//			printf("B/T=%d,%d iter=%d (done)\n", (int)blockIdx.x, (int)threadIdx.x, iter);

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
