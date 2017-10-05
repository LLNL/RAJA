#ifndef RAJA_pattern_nested_HPP
#define RAJA_pattern_nested_HPP


#include "RAJA/config.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/util/types.hpp"
#include "RAJA/policy/cuda.hpp"

#include "RAJA/pattern/nested/internal.hpp"

#include "camp/camp.hpp"
#include "camp/concepts.hpp"
#include "camp/tuple.hpp"

#include <iostream>
#include <type_traits>

namespace RAJA
{
namespace nested
{


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
  SegmentTuple st;
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

template <camp::idx_t Index, typename BaseWrapper>
struct GenericWrapper {
  using data_type = camp::decay<typename BaseWrapper::data_type>;
  GenericWrapper(BaseWrapper const &w) : bw{w} {}
  GenericWrapper(data_type &d) : bw{d} {}
  BaseWrapper bw;
};

template <camp::idx_t Index, typename BaseWrapper>
struct ForWrapper : GenericWrapper<Index, BaseWrapper> {
  using Base = GenericWrapper<Index, BaseWrapper>;
  using Base::Base;
  template <typename InIndexType>
  void operator()(InIndexType i)
  {
    Base::bw.data.template assign_index<Index>(i);
    Base::bw();
  }
};

template <typename T>
struct NestedPrivatizer {
  using data_type = typename T::data_type;
  using value_type = camp::decay<T>;
  using reference_type = value_type &;
  data_type data;
  value_type priv;
  NestedPrivatizer(const T &o) : data{o.bw.data}, priv{value_type{data}} {}
  reference_type get_priv() { return priv; }
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
    impl::forall(fp.pol,
                 camp::get<ForType::index_val>(wrap.data.st),
                 ForWrapper<ForType::index_val, WrappedBody>{wrap});
  }
};



template <typename ExecPolicy, typename... Fors>
struct Collapse : public internal::ForList, public internal::CollapseBase {
  using as_for_list = camp::list<Fors...>;
  const ExecPolicy pol;
  Collapse() : pol{} {}
  Collapse(ExecPolicy const &ep) : pol{ep} {}
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



template <int idx, int n_policies, typename Data, bool Own = false>
struct Wrapper {
  using Next = Wrapper<idx + 1, n_policies, Data>;
  using data_type = typename std::remove_reference<Data>::type;
  Data &data;
  explicit Wrapper(Data &d) : data{d} {}
  void operator()() const
  {
    auto const &pol = camp::get<idx>(data.pt);
    Executor<internal::remove_all_t<decltype(pol)>> e{};
    Next next_wrapper{data};
    e(pol, next_wrapper);
  }
};

// Innermost, execute body
template <int n_policies, typename Data, bool Own>
struct Wrapper<n_policies, n_policies, Data, Own> {
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


#include "RAJA/pattern/nested/tile.hpp"

#endif /* RAJA_pattern_nested_HPP */
