/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing tiling policies and mechanics
 *          for forallN templates.
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

#ifndef RAJA_pattern_nested_HPP
#define RAJA_pattern_nested_HPP


#include "RAJA/config.hpp"
#include "RAJA/policy/cuda.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/pattern/nested/internal.hpp"

#include "RAJA/util/chai_support.hpp"

#include "camp/camp.hpp"
#include "camp/concepts.hpp"
#include "camp/tuple.hpp"

#include <iostream>
#include <type_traits>

namespace RAJA
{
namespace nested
{
template <typename... Policies>
using Policy = camp::tuple<Policies...>;


template <camp::idx_t ArgumentId, typename ExecPolicy = camp::nil, typename... Rest>
struct For : public internal::ForList,
             public internal::ForTraitBase<ArgumentId, ExecPolicy> {
  using as_for_list = camp::list<For>;

  // used for execution space resolution
  using as_space_list = camp::list<For>;

  // TODO: add static_assert for valid policy in Pol
  const ExecPolicy exec_policy;
  RAJA_HOST_DEVICE constexpr For() : exec_policy{} {}
  RAJA_HOST_DEVICE constexpr For(const ExecPolicy &p) : exec_policy{p} {}
};


template <typename ExecPolicy, typename... Fors>
struct Collapse : public internal::ForList, public internal::CollapseBase {
  using as_for_list = camp::list<Fors...>;

  // used for execution space resolution
  using as_space_list = camp::list<For<-1, ExecPolicy>>;

  const ExecPolicy exec_policy;
  RAJA_HOST_DEVICE constexpr Collapse() : exec_policy{} {}
  RAJA_HOST_DEVICE constexpr Collapse(ExecPolicy const &ep) : exec_policy{ep} {}
};
}


#ifdef RAJA_ENABLE_CHAI

namespace detail
{


/*
 * Define CHAI support for nested policies.
 *
 * We need to walk the entire set of execution policies inside of the
 * RAJA::nested::Policy
 */
template <typename... POLICIES>
struct get_space<RAJA::nested::Policy<POLICIES...>>
    : public get_space_from_list<  // combines exec policies to find exec space

          // Extract just the execution policies from the tuple
          RAJA::nested::internal::get_space_policies<
              typename camp::tuple<POLICIES...>::TList>

          > {
};

}  // end detail namespace

#endif  // RAJA_ENABLE_CHAI


namespace nested
{


template <typename PolicyTuple, typename SegmentTuple, typename Body>
struct LoopData {

  constexpr static size_t n_policies = camp::tuple_size<PolicyTuple>::value;

  using index_tuple_t = internal::index_tuple_from_segments<
  typename SegmentTuple::TList>;


  const PolicyTuple policy_tuple;
  SegmentTuple segment_tuple;
  const typename std::remove_reference<Body>::type body;
  index_tuple_t index_tuple;


  LoopData(PolicyTuple const &p, SegmentTuple const &s, Body const &b)
      : policy_tuple(p), segment_tuple(s), body(b)
  {
  }
  template <camp::idx_t Idx, typename IndexT>
  RAJA_HOST_DEVICE void assign_index(IndexT const &i)
  {
    camp::get<Idx>(index_tuple) =
        camp::tuple_element_t<Idx, decltype(index_tuple)>{i};
  }
};


template <typename Policy>
struct Executor;

template <camp::idx_t Index, typename BaseWrapper>
struct GenericWrapper {
  using data_type = camp::decay<typename BaseWrapper::data_type>;

  BaseWrapper wrapper;

  GenericWrapper(BaseWrapper const &w) : wrapper{w} {}
  GenericWrapper(data_type &d) : wrapper{d} {}

};

template <camp::idx_t Index, typename BaseWrapper>
struct ForWrapper : GenericWrapper<Index, BaseWrapper> {
  using Base = GenericWrapper<Index, BaseWrapper>;
  using Base::Base;

  template <typename InIndexType>
  void operator()(InIndexType i)
  {
    Base::wrapper.data.template assign_index<Index>(i);
    Base::wrapper();
  }
};

template <typename T>
struct NestedPrivatizer {
  using data_type = typename T::data_type;
  using value_type = camp::decay<T>;
  using reference_type = value_type &;

  data_type privatized_data;
  value_type privatized_wrapper;

  NestedPrivatizer(const T &o) : privatized_data{o.wrapper.data}, privatized_wrapper{value_type{privatized_data}} {}

  reference_type get_priv() { return privatized_wrapper; }
};


/**
 * @brief specialization of internal::thread_privatize for nested
 */
template <camp::idx_t Index, typename BW>
auto thread_privatize(const nested::ForWrapper<Index, BW> &item)
    -> NestedPrivatizer<nested::ForWrapper<Index, BW>>
{
  return NestedPrivatizer<nested::ForWrapper<Index, BW>>{item};
}

template <typename ForType>
struct Executor {
  static_assert(std::is_base_of<internal::ForBase, ForType>::value,
                "Only For-based policies should get here");
  template <typename WrappedBody>
  void operator()(ForType const &fp, WrappedBody const &wrap)
  {
    using ::RAJA::policy::sequential::forall_impl;
    forall_impl(fp.exec_policy,
                camp::get<ForType::index_val>(wrap.data.segment_tuple),
                ForWrapper<ForType::index_val, WrappedBody>{wrap});
  }
};


//
// This is for demonstration only... can be removed eventually
//
template <typename FT0, typename FT1>
struct Executor<Collapse<seq_exec, FT0, FT1>> {
  static_assert(std::is_base_of<internal::ForBase, FT0>::value,
                "Only For-based policies should get here");
  static_assert(std::is_base_of<internal::ForBase, FT1>::value,
                "Only For-based policies should get here");
  template <typename WrappedBody>
  void operator()(Collapse<seq_exec, FT0, FT1> const &, WrappedBody const &wrap)
  {
    auto b0 = std::begin(camp::get<FT0::index_val>(wrap.data.segment_tuple));
    auto b1 = std::begin(camp::get<FT1::index_val>(wrap.data.segment_tuple));

    auto e0 = std::end(camp::get<FT0::index_val>(wrap.data.segment_tuple));
    auto e1 = std::end(camp::get<FT1::index_val>(wrap.data.segment_tuple));

    // Skip a level
    for (auto i0 = b0; i0 < e0; ++i0) {
      wrap.data.template assign_index<FT0::index_val>(*i0);
      for (auto i1 = b1; i1 < e1; ++i1) {
        wrap.data.template assign_index<FT1::index_val>(*i1);
        wrap();
      }
    }
  }
};



template <int idx, int n_policies, typename Data>
struct Wrapper {
  constexpr static int cur_policy = idx;
  constexpr static int num_policies = n_policies;
  using Next = Wrapper<idx + 1, n_policies, Data>;
  using data_type = typename std::remove_reference<Data>::type;
  Data &data;
  explicit Wrapper(Data &d) : data{d} {}
  void operator()() const
  {
    auto const &pol = camp::get<idx>(data.policy_tuple);
    Executor<internal::remove_all_t<decltype(pol)>> e{};
    Next next_wrapper{data};
    e(pol, next_wrapper);
  }
};

// Innermost, execute body
template <int n_policies, typename Data>
struct Wrapper<n_policies, n_policies, Data> {
  constexpr static int cur_policy = n_policies;
  constexpr static int num_policies = n_policies;
  using Next = Wrapper<n_policies, n_policies, Data>;
  using data_type = typename std::remove_reference<Data>::type;
  Data &data;
  explicit Wrapper(Data &d) : data{d} {}
  void operator()() const { camp::invoke(data.index_tuple, data.body); }
};



/**
 * Specialization of NestedPrivatizeer for Wrapper (used with Collapse policies)
 */
template <int idx, int n_policies, typename BW>
struct NestedPrivatizer<nested::Wrapper<idx, n_policies, BW>> {
  using data_type = typename nested::Wrapper<idx, n_policies, BW>::data_type;
  using value_type = nested::Wrapper<idx, n_policies, BW>;
  using reference_type = value_type &;
  data_type data;
  value_type priv;
  NestedPrivatizer(const nested::Wrapper<idx, n_policies, BW> &o) : data{o.data}, priv{value_type{data}} {}
  reference_type get_priv() { return priv; }
};


/**
 * @brief specialization of thread_privatize for nested Wrapper
 */
template <int idx, int n_policies, typename BW>
auto thread_privatize(const nested::Wrapper<idx, n_policies, BW> &item)
    -> NestedPrivatizer<nested::Wrapper<idx, n_policies, BW>>
{
  return NestedPrivatizer<Wrapper<idx, n_policies, BW>>{item};
}


template <typename Data>
auto make_base_wrapper(Data &d) -> Wrapper<0, Data::n_policies, Data>
{
  return Wrapper<0, Data::n_policies, Data>(d);
}

template <typename Pol, typename SegmentTuple, typename Body>
RAJA_INLINE void forall(const Pol &p, const SegmentTuple &st, const Body &b)
{
  detail::setChaiExecutionSpace<Pol>();

//  using fors = internal::get_for_policies<typename Pol::TList>;
  // TODO: ensure no duplicate indices in For<>s
  // TODO: ensure no gaps in For<>s
  // TODO: test that all policy members model the Executor policy concept
  // TODO: add a static_assert for functors which cannot be invoked with
  //       index_tuple
//  static_assert(camp::tuple_size<SegmentTuple>::value
//                    == camp::size<fors>::value,
//                "policy and segment index counts do not match");
  auto data = LoopData<Pol, SegmentTuple, Body>{p, st, b};
  auto ld = make_base_wrapper(data);
  // std::cout << typeid(ld).name() << std::endl
  //           << typeid(data.index_tuple).name() << std::endl;
  ld();

  detail::clearChaiExecutionSpace();
}

}  // end namespace nested

}  // end namespace RAJA



#endif /* RAJA_pattern_nested_HPP */
