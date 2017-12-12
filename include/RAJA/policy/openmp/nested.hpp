/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run forallN
 *
 ******************************************************************************
 */

#ifndef RAJA_policy_openmp_nested_HPP
#define RAJA_policy_openmp_nested_HPP

#include "RAJA/config.hpp"
#include "RAJA/pattern/nested.hpp"

#if defined(RAJA_ENABLE_OPENMP)

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

//#include "RAJA/policy/openmp/MemUtils_OPENMP.hpp"
#include "RAJA/policy/openmp/policy.hpp"

//#include "RAJA/internal/ForallNPolicy.hpp"
#include "RAJA/internal/LegacyCompatibility.hpp"


namespace RAJA
{
namespace nested
{

#if 0
template <template <camp::idx_t, typename...> class ForTypeIn,
          std::size_t block_size,
          camp::idx_t Index,
          typename... Rest>
struct Executor<ForTypeIn<Index, openmp_exec<block_size>, Rest...>> {
  using ForType = ForTypeIn<Index, openmp_exec<block_size>, Rest...>;
  static_assert(std::is_base_of<internal::ForBase, ForType>::value,
                "Only For-based policies should get here");
  template <typename BaseWrapper>
  struct ForWrapper {
    // Explicitly unwrap the data from the wrapper
    ForWrapper(BaseWrapper const &w) : data(w.data) {}
    using data_type = typename BaseWrapper::data_type;
    data_type data;
    template <typename InIndexType>
     void operator()(InIndexType i)
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
struct Executor<ForTypeIn<Index, openmp_loop_exec, Rest...>> {
  using ForType = ForTypeIn<Index, openmp_loop_exec, Rest...>;
  static_assert(std::is_base_of<internal::ForBase, ForType>::value,
                "Only For-based policies should get here");


  template <typename BaseWrapper>
  struct ForWrapper {
    // Explicitly unwrap the data from the wrapper
    ForWrapper(BaseWrapper const &w) : data(w.data) {}
    using data_type = typename BaseWrapper::data_type;
    data_type &data;
    template <typename InIndexType>
    void operator()(InIndexType i)
    {
      data.template assign_index<ForType::index_val>(i);
      camp::invoke(data.index_tuple, data.f);
    }
  };
  template <typename WrappedBody>
  void  operator()(ForType const &fp, WrappedBody const &wrap)
  {

    using ::RAJA::policy::openmp::forall_impl;
    forall_impl(fp.pol,
                camp::get<ForType::index_val>(wrap.data.st),
                ForWrapper<WrappedBody>{wrap});
  }
};


namespace internal
{


template <int idx, int n_policies, typename Data>
struct OpenmpWrapper {
  constexpr static int cur_policy = idx;
  constexpr static int num_policies = n_policies;
  using Next = OpenmpWrapper<idx + 1, n_policies, Data>;
  using data_type = typename std::remove_reference<Data>::type;
  Data &data;

  explicit   OpenmpWrapper(Data &d) : data{d} {}

  void   operator()() const
  {
    auto const &pol = camp::get<idx>(data.pt);
    Executor<internal::remove_all_t<decltype(pol)>> e{};
    Next next_wrapper{data};
    e(pol, next_wrapper);
  }
};

// Innermost, execute body
template <int n_policies, typename Data>
struct OpenmpWrapper<n_policies, n_policies, Data> {
  constexpr static int cur_policy = n_policies;
  constexpr static int num_policies = n_policies;
  using Next = OpenmpWrapper<n_policies, n_policies, Data>;
  using data_type = typename std::remove_reference<Data>::type;
  Data &data;

  explicit   OpenmpWrapper(Data &d) : data{d} {}

  void operator()() const
  {
    camp::invoke(data.index_tuple, data.f);
  }
};


}  // namespace internal

#endif

  struct omp_parallel_collapse_exec {
  };

  template <typename... FOR>
  using ompCollapse = Collapse<omp_parallel_collapse_exec, FOR...>;

  
  // TODO, check that FT... are openmp policies
  template <typename FT0, typename FT1>
  struct Executor<Collapse<omp_parallel_collapse_exec, FT0, FT1> > {
    
    static_assert(std::is_base_of<internal::ForBase, FT0>::value,
                  "Only For-based policies should get here");
    static_assert(std::is_base_of<internal::ForBase, FT1>::value,
                  "Only For-based policies should get here");
    template <typename WrappedBody>
    void operator()(Collapse<omp_parallel_collapse_exec, FT0, FT1> const &, WrappedBody const &wrap)
    {



      auto b0 = std::begin(camp::get<FT0::index_val>(wrap.data.st));
      auto b1 = std::begin(camp::get<FT1::index_val>(wrap.data.st));
      auto e0 = std::end(camp::get<FT0::index_val>(wrap.data.st));
      auto e1 = std::end(camp::get<FT1::index_val>(wrap.data.st));

#pragma omp parallel
      {
        auto privatizer = RAJA::nested::thread_privatize(wrap);
        auto private_wrap = privatizer.get_priv();
#pragma omp for collapse (2)
        for (auto i0 = b0; i0 < e0; ++i0) {        
          for (auto i1 = b1; i1 < e1; ++i1) {
            private_wrap.data.template assign_index<FT0::index_val>(*i0);
            private_wrap.data.template assign_index<FT1::index_val>(*i1);
            private_wrap();
          }
        }
      }
      

#if 0
      ptrdiff_t b0 = *std::begin(camp::get<FT0::index_val>(wrap.data.st));
      ptrdiff_t b1 = *std::begin(camp::get<FT1::index_val>(wrap.data.st));
      
      ptrdiff_t e0 = *std::end(camp::get<FT0::index_val>(wrap.data.st));
      ptrdiff_t e1 = *std::end(camp::get<FT1::index_val>(wrap.data.st));

#pragma omp parallel
      {
        //create thread-private loop data
        auto privatizer = RAJA::nested::thread_privatize(wrap);
        auto private_wrap = privatizer.get_priv();

#pragma omp for collapse (2)
        for (auto i0 = b0; i0 < e0; ++i0) {
          for (auto i1 = b1; i1 < e1; ++i1) {
            private_wrap.data.template assign_index<FT0::index_val>(i0);
            private_wrap.data.template assign_index<FT1::index_val>(i1);

            private_wrap();
          }
        }
      }

      auto b0 = std::begin(camp::get<FT0::index_val>(wrap.data.st));
      auto b1 = std::begin(camp::get<FT1::index_val>(wrap.data.st));
      auto e0 = std::end(camp::get<FT0::index_val>(wrap.data.st));
      auto e1 = std::end(camp::get<FT1::index_val>(wrap.data.st));

#pragma omp parallel
      {
        auto privatizer = RAJA::nested::thread_privatize(wrap);
        auto private_wrap = privatizer.get_priv();
#pragma omp for collapse (2)
        for (auto i0 = b0; i0 < e0; ++i0) {        
          for (auto i1 = b1; i1 < e1; ++i1) {
            private_wrap.data.template assign_index<FT0::index_val>(*i0);
            private_wrap.data.template assign_index<FT1::index_val>(*i1);
            private_wrap();
          }
        }
      }
#endif

     
 
    }
   
  };


}  // namespace nested
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_OPENMP guard

#endif  // closing endif for header file include guard
