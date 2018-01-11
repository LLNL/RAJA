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

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#ifndef RAJA_policy_cuda_nested_HPP
#define RAJA_policy_cuda_nested_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "RAJA/pattern/nested.hpp"

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
      camp::invoke(data.index_tuple, data.body);
    }
  };
  template <typename WrappedBody>
  void operator()(ForType const &fp, WrappedBody const &wrap)
  {

    using ::RAJA::policy::sequential::forall_impl;
    forall_impl(fp.exec_policy,
                camp::get<ForType::index_val>(wrap.data.segment_tuple),
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
      camp::invoke(data.index_tuple, data.body);
    }
  };
  template <typename WrappedBody>
  void RAJA_DEVICE operator()(ForType const &fp, WrappedBody const &wrap)
  {

    using ::RAJA::policy::cuda::forall_impl;
    forall_impl(fp.exec_policy,
                camp::get<ForType::index_val>(wrap.data.segment_tuple),
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
    auto const &pol = camp::get<idx>(data.policy_tuple);
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
    camp::invoke(data.index_tuple, data.body);
  }
};


}  // namespace internal


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
      auto loop_value = *(begin + policy());

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
  };


  template <typename WrappedBody>
  void operator()(collapse_policy const &, WrappedBody const &wrap)
  {
    CudaDim dims;

    /* As we create a cuda wrapper, we construct all of the cuda loop policies,
     * like cuda_thread_x_exec, their associated segment from wrap.data.segment_tuple
     *
     * This construction modifies the CudaDim to specify the correct number
     * of threads and blocks for the kernel launch
     *
     * The wrapped body is the device function to be launched, and does all
     * of the block/thread idx unpacking and assignment
    */
    auto begin_tuple = camp::make_tuple(
        camp::get<FOR_TYPES::index_val>(wrap.data.segment_tuple).begin()...);

    auto cuda_wrap =
        ForWrapper<WrappedBody,
                   decltype(begin_tuple),
                   typename FOR_TYPES::policy_type::cuda_exec_policy...>(
            wrap,

            begin_tuple,

            typename FOR_TYPES::policy_type::cuda_exec_policy(
                dims, camp::get<FOR_TYPES::index_val>(wrap.data.segment_tuple))...

            );


    // Only launch a kernel if we have at least one thing to do
    if (numBlocks(dims) > 0 && numThreads(dims) > 0) {

      cudaStream_t stream = 0;

      internal::cudaLauncher<<<dims.num_blocks, dims.num_threads, 0, stream>>>(
          RAJA::cuda::make_launch_body(
              dims.num_blocks, dims.num_threads, 0, stream, cuda_wrap));
      RAJA::cuda::peekAtLastError();

      RAJA::cuda::launch(stream);
      if (!Async) RAJA::cuda::synchronize(stream);
    }
  }
};


}  // namespace nested
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
