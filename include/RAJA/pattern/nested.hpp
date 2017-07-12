#ifndef RAJA_pattern_nested_HPP
#define RAJA_pattern_nested_HPP

#include "RAJA/RAJA.hpp"
#include "RAJA/config.hpp"
#include "RAJA/internal/tuple.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/external/metal.hpp"

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
struct ForBase {
};

using is_for_policy = metal::
    bind<metal::trait<std::is_base_of>, metal::always<ForBase>, metal::_1>;
}

template <typename Seq>
using get_for_policies = metal::remove_if<
    Seq,
    metal::bind<metal::lambda<metal::not_>, internal::is_for_policy>>;

template <typename ForPolicy>
using get_for_index_pair = typename ForPolicy::index_pair;

template <typename PairL, typename PairR>
using order_by_index =
    metal::number<(metal::first<PairL>::value < metal::first<PairR>::value)>;

template <typename Seq>
using get_for_policies_as_pairs =
    metal::transform<metal::lambda<get_for_index_pair>, get_for_policies<Seq>>;

template <typename Seq>
using get_for_index_types = metal::transform<
    metal::lambda<metal::second>,
    metal::sort<get_for_policies_as_pairs<Seq>, metal::lambda<order_by_index>>>;

template <typename Seq>
using index_tuple_from_policies =
    metal::apply<metal::lambda<RAJA::util::tuple>, get_for_index_types<Seq>>;

template <typename Pol,
          metal::int_ ArgumentId,
          typename IndexType = RAJA::Index_type>
struct For : public internal::ForBase {
  using index_type = IndexType;
  using index_pair = metal::pair<metal::number<ArgumentId>, IndexType>;
  using index = metal::number<ArgumentId>;
  const Pol pol;
  For() : pol{} {}
  For(const Pol &p) : pol{p} {}
};

template <typename... Policies>
using Policy = RAJA::util::tuple<Policies...>;

template <typename Policy>
struct Executor;

template <typename Pol, metal::int_ ArgumentId, typename IndexType>
struct Executor<For<Pol, ArgumentId, IndexType>> {
  using ForPol = For<Pol, ArgumentId, IndexType>;
  template <typename BaseWrapper>
  struct ForWrapper
  {
    ForWrapper(BaseWrapper const &w) : bw{w} {}
    BaseWrapper bw;
    template <typename InIndexType>
    void operator()(InIndexType i)
    {
      RAJA::util::get<ArgumentId>(bw.data.index_tuple) = IndexType{i};
      bw();
    }
  };
  template <typename WrappedBody>
  void operator()(ForPol const &fp, WrappedBody const &wrap)
  {
    impl::forall(fp.pol, RAJA::util::get<ArgumentId>(wrap.data.st), ForWrapper<WrappedBody>{wrap});
  }
};

template <typename PolicyTuple, typename SegmentTuple, typename Fn>
struct LoopData {
  constexpr static size_t n_policies =
      RAJA::util::tuple_size<PolicyTuple>::value;
  const PolicyTuple &pt;
  const SegmentTuple &st;
  const typename std::remove_reference<Fn>::type f;
  index_tuple_from_policies<typename PolicyTuple::TList> index_tuple;
  LoopData(PolicyTuple const &p, SegmentTuple const &s, Fn const &fn)
      : pt{p}, st{s}, f{fn}
  {
  }
};

template <int idx, int n_policies, typename Data>
struct Wrapper {
  using Next = Wrapper<idx + 1, n_policies, Data>;
  Data &data;
  explicit Wrapper(Data &d) : data{d} {}
  void operator()()
  {
    auto const &pol = RAJA::util::get<idx>(data.pt);
    Executor<internal::remove_all_t<decltype(pol)>> e{};
    e(pol, Next{data});
  }
};

// Innermost, execute body
template <int n_policies, typename Data>
struct Wrapper<n_policies, n_policies, Data> {
  Data &data;
  explicit Wrapper(Data &d) : data{d} {}
  void operator()() { RAJA::util::invoke(data.index_tuple, data.f); }
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
  using fors = get_for_policies<typename Pol::TList>;
  static_assert(RAJA::util::tuple_size<SegmentTuple>::value
                    == metal::size<fors>::value,
                "policy and segment index counts do not match");
  auto data = LoopData<Pol, SegmentTuple, Body>{p, st, b};
  auto ld = make_base_wrapper(data);
  std::cout << typeid(ld).name() << std::endl;
  ld();

#ifdef RAJA_ENABLE_CUDA
  afterCudaKernelLaunch();
#endif
}

}  // end namespace nested
}  // end namespace RAJA

#endif /* RAJA_pattern_nested_HPP */
