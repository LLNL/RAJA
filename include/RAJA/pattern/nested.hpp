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

template <camp::idx_t BodyIdx>
struct Lambda{
  using inside_policy_t = Policy<camp::nil>;

  inside_policy_t inside_policy;
};


template <camp::idx_t ArgumentId, typename ExecPolicy = camp::nil, typename... Rest>
struct For : public internal::ForList,
             public internal::ForTraitBase<ArgumentId, ExecPolicy> {
  using as_for_list = camp::list<For>;

  // used for execution space resolution
  using as_space_list = camp::list<For>;

  using inside_policy_t = Policy<Rest...>;

  // TODO: add static_assert for valid policy in Pol
  const ExecPolicy exec_policy;
  const inside_policy_t inside_policy;
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


template <typename PolicyTuple, typename SegmentTuple, typename ... Bodies>
struct LoopData {

  constexpr static size_t n_policies = camp::tuple_size<PolicyTuple>::value;

  using index_tuple_t = internal::index_tuple_from_segments<
  typename SegmentTuple::TList>;


  const PolicyTuple policy_tuple;
  SegmentTuple segment_tuple;

  using BodiesTuple = camp::tuple<typename std::remove_reference<Bodies>::type...> ;
  const BodiesTuple bodies;
  index_tuple_t index_tuple;


  LoopData(PolicyTuple const &p, SegmentTuple const &s, Bodies const & ... b)
      : policy_tuple(p), segment_tuple(s), bodies(b...)
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



template <camp::idx_t LoopIndex>
struct Executor<Lambda<LoopIndex>>{

  template <typename WrappedBody>
  void operator()(Lambda<LoopIndex>, WrappedBody const &wrap)
  {
    camp::invoke(wrap.data.index_tuple, camp::get<LoopIndex>(wrap.data.bodies));
  }
};


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


template <typename Policy, typename PolicyTuple, typename SegmentTuple, typename ... Bodies>
auto create_inside_data(Policy const &policy, LoopData<PolicyTuple, SegmentTuple, Bodies...> &data) ->
  LoopData<typename Policy::inside_policy_t, SegmentTuple, Bodies...>
{
  return LoopData<typename Policy::inside_policy_t, SegmentTuple, Bodies...>(policy, data.segment_tuple, data.bodies);
}




template <typename PolicyTuple, typename Data>
struct Wrapper {

  PolicyTuple const &policy_tuple;

  using policy_tuple_type = camp::decay<PolicyTuple>;
  constexpr static size_t num_seq_policies = camp::tuple_size<policy_tuple_type>::value;

  using data_type = typename std::remove_reference<Data>::type;
  Data &data;

  template <camp::idx_t idx, camp::idx_t N>
  struct SequentialExecutorCaller{

    RAJA_INLINE
    void operator()(PolicyTuple const &policy_tuple, Data &data) const {
//      printf("Wrapper seq<%d,%d>\n", (int)idx, (int)N);
      // Get the idx'th policy
      auto const &pol = camp::get<idx>(policy_tuple);

      // Create a wrapper for inside this policy
      Wrapper<decltype(pol.inside_policy), Data> inside_wrap(pol.inside_policy, data);

      // Create and call an Executor to do the work.
      Executor<internal::remove_all_t<decltype(pol)>> e{};
      e(pol, inside_wrap);


      // Call our next policy
      SequentialExecutorCaller<idx+1, N> next_sequential;
      next_sequential(policy_tuple, data);
    }
  };


  /*
   * termination case.
   * If any policies were specified, this becomes a NOP.
   * If no policies were specified, this defaults to Lambda<0>
   */

  template <camp::idx_t N>
  struct SequentialExecutorCaller<N,N> {

    RAJA_INLINE
    void operator()(PolicyTuple const &, Data &data) const {
      if(N == 0){
        camp::invoke(data.index_tuple, camp::get<0>(data.bodies));
      }
    }

  };


  Wrapper(PolicyTuple const &pt, Data &d) : policy_tuple(pt), data{d} {}

  void operator()() const
  {

    // Walk through policies in order

    SequentialExecutorCaller<0, num_seq_policies> launcher;
    launcher(policy_tuple, data);

  }
};



template <typename Data>
auto make_base_wrapper(Data &d) -> Wrapper<decltype(d.policy_tuple), Data>
{
  return Wrapper<decltype(d.policy_tuple), Data>(d.policy_tuple, d);
}


template <typename Pol, typename SegmentTuple, typename ... Bodies>
RAJA_INLINE void forall(const Pol &p, const SegmentTuple &st, const Bodies & ... b)
{
  detail::setChaiExecutionSpace<Pol>();

  // TODO: ensure no duplicate indices in For<>s
  // TODO: ensure no gaps in For<>s
  // TODO: test that all policy members model the Executor policy concept
  // TODO: add a static_assert for functors which cannot be invoked with
  //       index_tuple

  auto data = LoopData<Pol, SegmentTuple, Bodies...>(p, st, b...);
  auto ld = make_base_wrapper(data);
  // std::cout << typeid(ld).name() << std::endl
  //           << typeid(data.index_tuple).name() << std::endl;

  ld();

  detail::clearChaiExecutionSpace();
}

}  // end namespace nested


}  // end namespace RAJA



#endif /* RAJA_pattern_nested_HPP */
