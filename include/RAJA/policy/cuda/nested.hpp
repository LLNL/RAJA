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


namespace internal {



/*!
 * \brief Launcher that uses execution policies to map blockIdx and threadIdx to
 * map
 * to N-argument function
 */
template <typename BODY, typename... CARGS>
__global__ void cudaLauncher(BODY loop_body, CARGS... cargs)
{
  // force reduction object copy constructors and destructors to run
  auto body = loop_body;

  // Invoke the wrapped body.
  // The wrapper will take care of computing indices, and deciding if the
  // given block+thread is in-bounds, and invoking the users loop body
  body();
}


} // namespace internal


template<bool Async = false>
struct cuda_collapse_exec : public cuda_exec<0, Async>
{};


template<typename ... FOR>
using CudaCollapse = Collapse<cuda_collapse_exec<false>, FOR ...>;

template<typename ... FOR>
using CudaCollapseAsync = Collapse<cuda_collapse_exec<true>, FOR ...>;


// TODO, check that FT... are cuda policies
template <bool Async, typename ... FT>
struct Executor<Collapse<cuda_collapse_exec<Async>, FT ...>> {

  using collapse_policy = Collapse<cuda_collapse_exec<Async>, FT...>;



  template <typename BaseWrapper, typename ... LoopPol>
  struct ForWrapper {
    using data_type = typename BaseWrapper::data_type;
    data_type data;
    camp::tuple<LoopPol...> loop_policies;
    using ft_tuple = camp::list<FT...>;
    using index_sequence = typename camp::make_idx_seq<sizeof...(LoopPol)>::type;

    // Explicitly unwrap the data from the wrapper
    ForWrapper(BaseWrapper const &w, LoopPol &&... pol) :
      data(w.data),
      loop_policies(camp::make_tuple(std::forward<LoopPol>(pol)...))
    {}


    /*
     * Evaluates the loop index for the idx'th loop in the Collapse
     */
    template<size_t idx>
    RAJA_DEVICE
    RAJA::Index_type evalLoopIndex(){
      // grab the loop policy
      auto &policy = camp::get<idx>(loop_policies);

      // grab the For type from our type list
      using ft = typename camp::at_v<ft_tuple, idx>::type;

      // Assign the For index value to the correct argument
      int loop_value = policy();
      data.template assign_index<ft::index_val>( loop_value );

      return loop_value;
    }

    /*
     * Computes all of the loop indices, and returns true if the current
     * thread is valid (in-bounds)
     *
     * Since we use INT_MIN as a sentinel to mark out-of-bounds, the minimum
     * loop index must be > INT_MIN for this to be a valid thread.
     */
    template<camp::idx_t ... idx_list>
    RAJA_DEVICE
    bool computeIndices(camp::idx_seq<idx_list...>){
      // Compute each loop index, and return the minimum value
      return INT_MIN < VarOps::min<RAJA::Index_type>(evalLoopIndex<idx_list>()...);
    }


    RAJA_DEVICE void operator()()
    {
      // Assign the indices, and compute minimum loop index
      bool thread_valid =  computeIndices(index_sequence{});

      // Invoke the loop body, only if we are on a valid index
      // if any of the loops indices were out-of-bounds, then min_val will
      // be INT_MIN
      if(thread_valid){
        camp::invoke(data.index_tuple, data.f);
      }
    }
  };


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
    auto cuda_wrap =
        ForWrapper<WrappedBody, typename FT::policy_type::cuda_exec_policy...>(
            wrap,

            typename FT::policy_type::cuda_exec_policy (
                    dims, camp::get<FT::index_val>(wrap.data.st)
             ) ...);


    // Only launch a kernel if we have at least one thing to do
    if (numBlocks(dims) > 0 && numThreads(dims) > 0) {

      cudaStream_t stream = 0;

      internal::cudaLauncher<<<dims.num_blocks, dims.num_threads, 0, stream>>>(cuda_wrap);
      RAJA::cuda::peekAtLastError();

      RAJA::cuda::launch(stream);
      if (!Async) RAJA::cuda::synchronize(stream);
    }

  }
};



}  //namespace nested
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
