#ifndef RAJA_pattern_nested_HPP
#define RAJA_pattern_nested_HPP

#include "RAJA/RAJA.hpp"
#include "RAJA/config.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/util/types.hpp"

// #include "RAJA/external/metal.hpp"
#include "camp/camp.hpp"
#include "camp/tuple.hpp"

#include <iostream>
#include <type_traits>

namespace RAJA
{
namespace nested
{

namespace internal
{

template <typename T>
using remove_all_t =
    typename std::remove_cv<typename std::remove_reference<T>::type>::type;

// Universal base of all For wrappers for type traits
struct ForList {
};
struct ForBase {
};
struct TypedForBase : public ForBase {
};
struct CollapseBase {
};
template <camp::idx_t ArgumentId, typename Pol>
struct ForTraitBase : public ForBase {
  constexpr static camp::idx_t index_val = ArgumentId;
  using index = camp::num<ArgumentId>;
  using index_type = camp::nil;  // default to invalid type
  using policy_type = Pol;
  using type = ForTraitBase;  // make camp::value compatible
};

using is_for_policy = typename camp::bind_front<std::is_base_of, ForBase>::type;
using is_typed_for_policy =
    typename camp::bind_front<std::is_base_of, TypedForBase>::type;

using has_for_list = typename camp::bind_front<std::is_base_of, ForList>::type;

template <typename T>
using get_for_list = typename T::as_for_list;

template <typename Seq>
using get_for_policies = typename camp::flatten<typename camp::transform<
    get_for_list,
    typename camp::filter_l<has_for_list, Seq>::type>::type>::type;

template <typename T>
using is_nil_type = camp::bind_front<camp::concepts::metalib::is_same, camp::nil>;

template <typename Index, typename ForPol>
struct index_matches {
  using type = camp::num<Index::value == ForPol::index::value>;
};

template <typename IndexTypes,
          typename ForPolicies,
          typename Current,
          typename Index>
struct evaluate_policy {
  using ForPolicy = typename camp::find_if_l<
      typename camp::bind_front<index_matches, Index>::type,
      ForPolicies>::type;
  using type = typename camp::append<
      Current,
      camp::if_<typename std::is_base_of<TypedForBase, ForPolicy>::type,
                typename ForPolicy::index_type,
                typename camp::at<IndexTypes,
                                  typename ForPolicy::index>::type>>::type;
};

template <typename Policies, typename IndexTypes>
using get_for_index_types = typename camp::accumulate_l<
    typename camp::bind_front<evaluate_policy,
                              IndexTypes,
                              get_for_policies<Policies>>::type,
    camp::list<>,
    typename camp::as_list<camp::idx_seq_from_t<IndexTypes>>::type>::type;

template <typename Iterator>
struct iterable_value_type_getter {
  using type = typename Iterator::iterator::value_type;
};
template <>
struct iterable_value_type_getter<IndexSet> {
  // TODO: when static indexset drops, specialize properly
  using type = Index_type;
};

template <typename Segments>
using value_type_list_from_segments =
    typename camp::transform<iterable_value_type_getter, Segments>::type;

template <typename Policies, typename Segments>
using index_tuple_from_policies_and_segments = typename camp::apply_l<
    camp::lambda<camp::tuple>,
    get_for_index_types<Policies,
                        value_type_list_from_segments<Segments>>>::type;
}

template <camp::idx_t ArgumentId, typename Pol = camp::nil, typename... Rest>
struct For : public internal::ForList,
             public internal::ForTraitBase<ArgumentId, Pol> {
  using as_for_list = camp::list<For>;
  // TODO: add static_assert for valid policy in Pol
  const Pol pol;
  For() : pol{} {}
  For(const Pol &p) : pol{p} {}
};

template <camp::idx_t ArgumentId,
          typename Pol,
          typename IndexType,
          typename... Rest>
struct TypedFor : public internal::TypedForBase,
                  public For<ArgumentId, Pol, Rest...> {
  using Base = For<ArgumentId, Pol, Rest...>;
  using Self = TypedFor<ArgumentId, Pol, IndexType, Rest...>;
  using index_type = IndexType;
  using as_for_list = camp::list<Self>;
  // TODO: add static_assert for valid policy in Pol
  using Base::Base;
};

template <typename... Policies>
using Policy = camp::tuple<Policies...>;

template <typename PolicyTuple, typename SegmentTuple, typename Fn>
struct LoopData {
  constexpr static size_t n_policies = camp::tuple_size<PolicyTuple>::value;
  const PolicyTuple &pt;
  const SegmentTuple &st;
  const typename std::remove_reference<Fn>::type f;
  using index_tuple_t = internal::index_tuple_from_policies_and_segments<
      typename PolicyTuple::TList,
      typename SegmentTuple::TList>;
  index_tuple_t index_tuple;
  LoopData(PolicyTuple const &p, SegmentTuple const &s, Fn const &fn)
      : pt(p), st(s), f(fn)
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

template <typename ForType>
struct Executor {
  static_assert(std::is_base_of<internal::ForBase, ForType>::value,
                "Only For-based policies should get here");
  template <typename BaseWrapper>
  struct ForWrapper {
    ForWrapper(BaseWrapper const &w) : bw{w} {}
    BaseWrapper bw;
    template <typename InIndexType>
    void operator()(InIndexType i)
    {
      bw.data.template assign_index<ForType::index_val>(i);
      bw();
    }
  };
  template <typename WrappedBody>
  void operator()(ForType const &fp, WrappedBody const &wrap)
  {

    impl::forall(fp.pol,
                 camp::get<ForType::index_val>(wrap.data.st),
                 ForWrapper<WrappedBody>{wrap});
  }
};

#if defined(RAJA_ENABLE_CUDA)
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
    typename BaseWrapper::data_type data;
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

    impl::forall(fp.pol,
                 camp::get<ForType::index_val>(wrap.data.st),
                 ForWrapper<WrappedBody>{wrap});
  }
};
#endif

template <typename ExecPolicy, typename... Fors>
struct Collapse : public internal::ForList, public internal::CollapseBase {
  using as_for_list = camp::list<Fors...>;
  const ExecPolicy pol;
  Collapse() : pol{} {}
  Collapse(ExecPolicy const &ep) : pol{ep} {}
};

template <typename FT0, typename FT1>
struct Executor<Collapse<seq_exec, FT0, FT1>> {
  static_assert(std::is_base_of<internal::ForBase, FT0>::value,
                "Only For-based policies should get here");
  static_assert(std::is_base_of<internal::ForBase, FT1>::value,
                "Only For-based policies should get here");
  template <typename WrappedBody>
  void operator()(Collapse<seq_exec, FT0, FT1> const &, WrappedBody const &wrap)
  {
    auto b0 = std::begin(camp::get<FT0::index_val>(wrap.data.st));
    auto b1 = std::begin(camp::get<FT1::index_val>(wrap.data.st));

    auto e0 = std::end(camp::get<FT0::index_val>(wrap.data.st));
    auto e1 = std::end(camp::get<FT1::index_val>(wrap.data.st));

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
  using Next = Wrapper<idx + 1, n_policies, Data>;
  using data_type = typename std::remove_reference<Data>::type;
  Data &data;
  explicit Wrapper(Data &d) : data{d} {}
  void operator()() const
  {
    auto const &pol = camp::get<idx>(data.pt);
    Executor<internal::remove_all_t<decltype(pol)>> e{};
    e(pol, Next{data});
  }
};

// Innermost, execute body
template <int n_policies, typename Data>
struct Wrapper<n_policies, n_policies, Data> {
  using data_type = typename std::remove_reference<Data>::type;
  Data &data;
  explicit Wrapper(Data &d) : data{d} {}
  void operator()() const { camp::invoke(data.index_tuple, data.f); }
};

template <typename Data>
auto make_base_wrapper(Data &d) -> Wrapper<0, Data::n_policies, Data>
{
  return Wrapper<0, Data::n_policies, Data>(d);
}

template <typename Pol, typename SegmentTuple, typename Body>
RAJA_INLINE void forall(const Pol &p, const SegmentTuple &st, const Body &b)
{
#ifdef RAJA_ENABLE_CUDA
  // this call should be moved into a cuda file
  // but must be made before loop_body is copied
  beforeCudaKernelLaunch();
#endif
  using fors = internal::get_for_policies<typename Pol::TList>;
  // TODO: ensure no duplicate indices in For<>s
  // TODO: ensure no gaps in For<>s
  // TODO: test that all policy members model the Executor policy concept
  // TODO: add a static_assert for functors which cannot be invoked with
  //       index_tuple
  static_assert(camp::tuple_size<SegmentTuple>::value
                    == camp::size<fors>::value,
                "policy and segment index counts do not match");
  auto data = LoopData<Pol, SegmentTuple, Body>{p, st, b};
  auto ld = make_base_wrapper(data);
  // std::cout << typeid(ld).name() << std::endl
  //           << typeid(data.index_tuple).name() << std::endl;
  ld();

#ifdef RAJA_ENABLE_CUDA
  afterCudaKernelLaunch();
#endif
}

}  // end namespace nested
}  // end namespace RAJA

#endif /* RAJA_pattern_nested_HPP */
