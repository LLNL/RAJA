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

#ifndef RAJA_policy_cuda_nested_HPP
#define RAJA_policy_cuda_nested_HPP

#include "RAJA/config.hpp"
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

#include "RAJA/RAJA.hpp"
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


template <template <camp::idx_t, typename...> class ForTypeIn,
          std::size_t block_size,
          camp::idx_t Index,
          typename... Rest>
struct Executor<ForTypeIn<Index, cuda_exec<block_size>, Rest...>> {
  using ForType = ForTypeIn<Index, cuda_exec<block_size>, Rest...>;
  static_assert(std::is_base_of<internal::ForBase, ForType>::value,
                "Only For-based policies should get here");
  template <typename BaseWrapper>
  struct ForWrapper {
    // Explicitly unwrap the data from the wrapper
    ForWrapper(BaseWrapper const &w) : data(w.data) {}
    using data_type = typename BaseWrapper::data_type;
    data_type data;
    template <typename InIndexType>
    RAJA_DEVICE void operator()(InIndexType i)
    {
      data.template assign_index<ForType::index_val>(i);
      camp::invoke(data.index_tuple, data.f);
    }
  };
  template <typename WrappedBody>
  void operator()(ForType const &fp, WrappedBody const &wrap)
  {

    using ::RAJA::policy::sequential::forall_impl;
    forall_impl(fp.pol,
                camp::get<ForType::index_val>(wrap.data.st),
                ForWrapper<WrappedBody>{wrap});
  }
};


template <template <camp::idx_t, typename...> class ForTypeIn,
          camp::idx_t Index,
          typename... Rest>
struct Executor<ForTypeIn<Index, cuda_loop_exec, Rest...>> {
  using ForType = ForTypeIn<Index, cuda_loop_exec, Rest...>;
  static_assert(std::is_base_of<internal::ForBase, ForType>::value,
                "Only For-based policies should get here");


  template <typename BaseWrapper>
  struct ForWrapper {
    // Explicitly unwrap the data from the wrapper
    RAJA_DEVICE ForWrapper(BaseWrapper const &w) : data(w.data) {}
    using data_type = typename BaseWrapper::data_type;
    data_type &data;
    template <typename InIndexType>
    RAJA_DEVICE void operator()(InIndexType i)
    {
      data.template assign_index<ForType::index_val>(i);
      camp::invoke(data.index_tuple, data.f);
    }
  };
  template <typename WrappedBody>
  void RAJA_DEVICE operator()(ForType const &fp, WrappedBody const &wrap)
  {

    using ::RAJA::policy::cuda::forall_impl;
    forall_impl(fp.pol,
                camp::get<ForType::index_val>(wrap.data.st),
                ForWrapper<WrappedBody>{wrap});
  }
};


namespace internal
{


/*!
 * \brief Launcher that uses execution policies to map blockIdx and threadIdx to
 * map
 * to N-argument function
 */
template <typename BODY>
__global__ void cudaLauncher(BODY loop_body)
{
  // force reduction object copy constructors and destructors to run
  auto body = loop_body;

  // Invoke the wrapped body.
  // The wrapper will take care of computing indices, and deciding if the
  // given block+thread is in-bounds, and invoking the users loop body
  body();
}


template <int idx, int n_policies, typename Data>
struct CudaWrapper {
  constexpr static int cur_policy = idx;
  constexpr static int num_policies = n_policies;
  using Next = CudaWrapper<idx + 1, n_policies, Data>;
  using data_type = typename std::remove_reference<Data>::type;
  Data &data;

  explicit RAJA_DEVICE CudaWrapper(Data &d) : data{d} {}

  void RAJA_DEVICE operator()() const
  {
    auto const &pol = camp::get<idx>(data.pt);
    Executor<internal::remove_all_t<decltype(pol)>> e{};
    Next next_wrapper{data};
    e(pol, next_wrapper);
  }
};

// Innermost, execute body
template <int n_policies, typename Data>
struct CudaWrapper<n_policies, n_policies, Data> {
  constexpr static int cur_policy = n_policies;
  constexpr static int num_policies = n_policies;
  using Next = CudaWrapper<n_policies, n_policies, Data>;
  using data_type = typename std::remove_reference<Data>::type;
  Data &data;

  explicit RAJA_DEVICE CudaWrapper(Data &d) : data{d} {}

  void RAJA_DEVICE operator()() const
  {
    camp::invoke(data.index_tuple, data.f);
  }
};


}  // namespace internal

template <typename Data>
auto make_cuda_wrapper(Data &d) -> internal::CudaWrapper<1, Data::n_policies, Data>
{
  return internal::CudaWrapper<1, Data::n_policies, Data>(d);
}


template <bool Async = false>
struct cuda_collapse_exec : public cuda_exec<0, Async> {
};


template <typename... FOR>
using CudaCollapse = Collapse<cuda_collapse_exec<false>, FOR...>;

template <typename... FOR>
using CudaCollapseAsync = Collapse<cuda_collapse_exec<true>, FOR...>;


// TODO, check that FT... are cuda policies
template <bool Async, typename... FOR_TYPES>
struct Executor<Collapse<cuda_collapse_exec<Async>, FOR_TYPES...>> {

  using collapse_policy = Collapse<cuda_collapse_exec<Async>, FOR_TYPES...>;


  template <typename BaseWrapper, typename BeginTuple, typename... LoopPol>
  struct ForWrapper {

    using data_type = typename BaseWrapper::data_type;
    data_type data;

    BeginTuple begin_tuple;

    using CuWrap = internal::CudaWrapper<BaseWrapper::cur_policy,
                                         BaseWrapper::num_policies,
                                         typename BaseWrapper::data_type>;

    camp::tuple<LoopPol...> loop_policies;

    using ft_tuple = camp::list<FOR_TYPES...>;


    // Explicitly unwrap the data from the wrapper
    ForWrapper(BaseWrapper const &w,
               BeginTuple const &bt,
               LoopPol const &... pol)
        : data(w.data), begin_tuple(bt), loop_policies(camp::make_tuple(pol...))
    {
    }


    /*
     * Evaluates the loop index for the idx'th loop in the Collapse
     *
     * returns false if the loop is in bounds
     */
    template <size_t idx>
    RAJA_DEVICE bool evalLoopIndex()
    {
      // grab the loop policy
      auto &policy = camp::get<idx>(loop_policies);

      // grab the For type from our type list
      using ft = typename camp::at_v<ft_tuple, idx>::type;


      // Get the offset for this policy, calculated by our thread/block index
      RAJA::Index_type offset = policy();
      if(offset == RAJA::operators::limits<RAJA::Index_type>::min()){
        // this index is out-of-bounds, so shortcut it here
        return true;
      }

      // Increment the begin iterator by the offset
      auto &begin = camp::get<idx>(begin_tuple);
      RAJA::Index_type loop_value = *(begin + policy());

      // Assign the For index value to the correct argument
      data.template assign_index<ft::index_val>(loop_value);

      // we are in bounds
      return false;
    }

    /*
     * Computes all of the loop indices, and returns true if the current
     * thread is valid (in-bounds)
     *
     * Since we use INT_MIN as a sentinel to mark out-of-bounds, the minimum
     * loop index must be > INT_MIN for this to be a valid thread.
     */
    template <camp::idx_t... idx_list>
    RAJA_DEVICE bool computeIndices(camp::idx_seq<idx_list...>)
    {
      // Compute each loop index, and return the minimum value
      // this sum is the number of indices that are out of bounds
      return 0 == VarOps::sum<int>((int)evalLoopIndex<idx_list>()...);
    }


    RAJA_DEVICE void operator()()
    {
      // Assign the indices, and compute minimum loop index
      using index_sequence =
          typename camp::make_idx_seq<sizeof...(LoopPol)>::type;
      bool in_bounds = computeIndices(index_sequence{});

      // Invoke the loop body, only if we are on a valid index
      // if any of the loops indices were out-of-bounds, then min_val will
      // be INT_MIN
      if (in_bounds) {
        CuWrap wrap(data);
        wrap();
      }
    }

    RAJA_DEVICE void operator()() const {
      printf("you got const!\n");
    }
  };


  template<typename Segments>
  CudaDim computeCudaDim(Segments const &segment_tuple) const {

    CudaDim dims;

    VarOps::ignore_args(typename FOR_TYPES::policy_type::cuda_exec_policy(
                dims, camp::get<FOR_TYPES::index_val>(segment_tuple))...);

    return dims;
  }

//  template <typename WrappedBody, typename BeginTuple>
//  RAJA_INLINE
//  auto createDeviceWrapper(
//      CudaDim &dims,
//      BeginTuple const &begin_tuple,
//      WrappedBody const &wrap) ->
//
//      ForWrapper<WrappedBody,
//                  BeginTuple,
//                 typename FOR_TYPES::policy_type::cuda_exec_policy...>
//
//  {
//
//    /* As we create a cuda wrapper, we construct all of the cuda loop policies,
//     * like cuda_thread_x_exec, their associated segment from wrap.data.st
//     *
//     * This construction modifies the CudaDim to specify the correct number
//     * of threads and blocks for the kernel launch
//     *
//     * The wrapped body is the device function to be launched, and does all
//     * of the block/thread idx unpacking and assignment
//    */
//    return
//        ForWrapper<WrappedBody,
//                   BeginTuple,
//                   typename FOR_TYPES::policy_type::cuda_exec_policy...>(
//            wrap,
//
//            begin_tuple,
//
//            typename FOR_TYPES::policy_type::cuda_exec_policy(
//                dims, camp::get<FOR_TYPES::index_val>(wrap.data.st))...
//
//            );
//
//  }


  template<typename WrappedBody>
  using BeginTuple = camp::tuple<typename camp::tuple_element_t<FOR_TYPES::index_val, typename WrappedBody::data_type::segment_tuple_type>::iterator...>;

  template<typename WrappedBody>
  using DeviceWrapper = ForWrapper<WrappedBody,
      BeginTuple<WrappedBody>,
      typename FOR_TYPES::policy_type::cuda_exec_policy...>;

  template <typename WrappedBody>
  RAJA_INLINE
  DeviceWrapper<WrappedBody> createDeviceWrapper(
      CudaDim &dims,
      WrappedBody const &wrap)
  {

      /* As we create a cuda wrapper, we construct all of the cuda loop policies,
       * like cuda_thread_x_exec, their associated segment from wrap.data.st
       *
       * This construction modifies the CudaDim to specify the correct number
       * of threads and blocks for the kernel launch
       *
       * The wrapped body is the device function to be launched, and does all
       * of the block/thread idx unpacking and assignment
      */
      return
          DeviceWrapper<WrappedBody>(
              wrap,

              BeginTuple<WrappedBody>{camp::get<FOR_TYPES::index_val>(wrap.data.st).begin()...},

              typename FOR_TYPES::policy_type::cuda_exec_policy(
                  dims, camp::get<FOR_TYPES::index_val>(wrap.data.st))...

              );

    }


  template <typename WrappedBody>
  void operator()(collapse_policy const &, WrappedBody const &wrap)
  {
    CudaDim dims;

    /* As we create a cuda wrapper, we construct all of the cuda loop policies,
     * like cuda_thread_x_exec, their associated segment from wrap.data.st
     *
     * This construction modifies the CudaDim to specify the correct number
     * of threads and blocks for the kernel launch
     *
     * The wrapped body is the device function to be launched, and does all
     * of the block/thread idx unpacking and assignment
    */

    auto cuda_wrap = createDeviceWrapper(dims, wrap);

    // Only launch a kernel if we have at least one thing to do
    if (numBlocks(dims) > 0 && numThreads(dims) > 0) {

      cudaStream_t stream = 0;

      // Get amount of dynamic shared memory requested by SharedMemory objects
      size_t shmem = RAJA::cuda::detail::shared_memory_total_bytes;

      internal::cudaLauncher<<<dims.num_blocks, dims.num_threads,
          shmem, stream>>>(
          RAJA::cuda::make_launch_body(
              dims.num_blocks, dims.num_threads, shmem, stream, cuda_wrap));
      RAJA::cuda::peekAtLastError();

      RAJA::cuda::launch(stream);
      if (!Async) RAJA::cuda::synchronize(stream);
    }
  }
};




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
  size_t shmem = RAJA::cuda::detail::shared_memory_total_bytes;
  printf("Dynamic shared memory: %ld bytes\n", (long)shmem);
  shmem = 4096;  // TODO: FIXME!!!


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
