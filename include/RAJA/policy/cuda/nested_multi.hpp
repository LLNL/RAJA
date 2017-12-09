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

#include "RAJA/pattern/nested.hpp"
#include "RAJA/pattern/nested_multi.hpp"

#ifndef RAJA_policy_cuda_nested__multi_HPP
#define RAJA_policy_cuda_nested__multi_HPP

#include "RAJA/config.hpp"


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



#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#include "RAJA/policy/cuda/policy.hpp"

#include "RAJA/internal/ForallNPolicy.hpp"
#include "RAJA/internal/LegacyCompatibility.hpp"


namespace RAJA
{
namespace nested
{




template<bool Async, size_t i, size_t N>
struct InvokeLoopsCUDA {

  template<typename ... LoopDims, typename ... LoopList>
  RAJA_DEVICE
  void operator()(camp::tuple<LoopDims...> const &loop_dims,
                  camp::tuple<LoopList...> &loops) {

    if(camp::get<i>(loop_dims).threadIncluded()){
      //invokeLoopData(camp::get<i>(loops));
      camp::get<i>(loops)();
    }

    if(!Async){
      __syncthreads();
    }

    InvokeLoopsCUDA<Async, i+1, N> next_invoke;
    next_invoke(loop_dims, loops);
  }

};


template<bool Async, size_t N>
struct InvokeLoopsCUDA<Async, N, N> {

  template<typename ... LoopDims, typename ... LoopList>
  RAJA_DEVICE
  void operator()(camp::tuple<LoopDims...> const &,
      camp::tuple<LoopList...> &) {}

};




template<bool Async = false>
struct cuda_multi_exec{};


template<typename NestedPolicy, typename SegmentTuple, typename Body>
auto createLoopExecutor(LoopData<NestedPolicy, SegmentTuple, Body> const &loop_data) ->
Executor<camp::at_v<NestedPolicy, 0>>
{
  // Extract the first policy from the RAJA::nested::Policy
  // We are assuming that this policy is going to be a CudaCollapse
  using collapse_policy = camp::at_v<NestedPolicy, 0>;

  // Use the Executor class to compute what thread/block dimensions
  // are needed for this kernel
  Executor<collapse_policy> exec;

  return exec;
}


template<typename NestedPolicy, typename SegmentTuple, typename Body>
CudaDim computeCudaDims(CudaDim &launch_dims,
    LoopData<NestedPolicy, SegmentTuple, Body> const &loop_data)
{
  // Extract the first policy from the RAJA::nested::Policy
  // We are assuming that this policy is going to be a CudaCollapse
  using collapse_policy = camp::at_v<NestedPolicy, 0>;

  // Use the Executor class to compute what thread/block dimensions
  // are needed for this kernel
  Executor<collapse_policy> exec;
  CudaDim dims = exec.computeCudaDim(loop_data.st);

  printf("Loop Dims: \n");
  dims.print();

  // keep track of maximum launch dimensions
  launch_dims = launch_dims.maximum(dims);

  // Return this kernels launch dims
  return dims;
}

template<bool Async, typename LoopDimList, typename LoopList>
struct CudaMultiWrapper {};

template<bool Async, typename ... LoopDims, typename ... Loops>
struct CudaMultiWrapper<Async, camp::tuple<LoopDims...>, camp::tuple<Loops...>>{
  camp::tuple<LoopDims...> loop_dims;
  camp::tuple<Loops...> loops;

  RAJA_DEVICE void operator()()
  {
    InvokeLoopsCUDA<Async, 0, sizeof...(Loops)> invoker;
    invoker(loop_dims, loops);
  }
};




template <bool Async, camp::idx_t... LoopIdx, typename ... LoopList>
RAJA_INLINE void forall_multi_idx(
    cuda_multi_exec<Async>,
    camp::idx_seq<LoopIdx...> const &,
    LoopList & ... loop_datas)
{

  // Create a tuple of Executor objects

  auto executors = camp::make_tuple(createLoopExecutor(loop_datas)...);



  auto loop_tuple = camp::make_tuple(make_cuda_wrapper(loop_datas)...);



  // Create a tuple of device wrappers from the executors
  // also, compute shared memory requirements
  RAJA::detail::startSharedMemorySetup();

  CudaDim foo_dim;
  auto loop_wraps = camp::make_tuple(
      camp::get<LoopIdx>(executors).createDeviceWrapper(
          foo_dim, camp::get<LoopIdx>(loop_tuple)
      )
      ...
  );

  RAJA::detail::finishSharedMemorySetup();



  // Compute dim3's for thread and blocks, for each loop
  // launch_dims becomes the max over all of the dimensions
  CudaDim dims;
  auto loop_dims = camp::make_tuple(computeCudaDims(dims, loop_datas)...);


  printf("Launch Dims: \n");
  dims.print();

  // Step 3: Wrap the loops with a device-side invoker

  using wrapper_type = CudaMultiWrapper<Async, decltype(loop_dims), decltype(loop_wraps)>;
  wrapper_type wrap {loop_dims, loop_wraps};


  // Step 4: launch our kernel!
  cudaStream_t stream = 0;

  // Get amount of dynamic shared memory requested by SharedMemory objects
  size_t shmem = RAJA::detail::getSharedMemorySize();
//  printf("Dynamic shared memory: %ld bytes\n", (long)shmem);


  internal::cudaLauncher<<<dims.num_blocks, dims.num_threads,
      shmem, stream>>>(
      RAJA::cuda::make_launch_body(
          dims.num_blocks, dims.num_threads, shmem, stream, wrap));
  RAJA::cuda::peekAtLastError();

  RAJA::cuda::launch(stream);
}

template <bool Async, typename ... LoopList>
RAJA_INLINE void forall_multi(
    cuda_multi_exec<Async> const &exec,
    LoopList ... loop_datas)
{

  using loop_idx = typename camp::make_idx_seq<sizeof...(LoopList)>::type;

    forall_multi_idx(exec, loop_idx{},  loop_datas...);

}


}  // namespace nested
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
