#ifndef RAJA_pattern_nested_HPP
#define RAJA_pattern_nested_HPP

#include "RAJA/RAJA.hpp"
#include "RAJA/config.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/util/types.hpp"

// #include "RAJA/external/metal.hpp"
#include "camp/camp.hpp"
#include "camp/concepts.hpp"
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
using is_nil_type =
    camp::bind_front<camp::concepts::metalib::is_same, camp::nil>;

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
    camp::as_list<camp::idx_seq_from_t<IndexTypes>>>::type;

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

template <camp::idx_t Index, typename TilePolicy, typename ExecPolicy>
struct Tile {
  const TilePolicy tpol;
  const ExecPolicy epol;
  Tile(TilePolicy const &tp = TilePolicy{}, ExecPolicy const &ep = ExecPolicy{})
      : tpol{tp}, epol{ep}
  {
  }
};

///! tag for a tiling loop
template <camp::idx_t chunk_size_>
struct tile_static {
  static constexpr camp::idx_t chunk_size = chunk_size_;

  tile_static() {}
  constexpr camp::idx_t get_chunk_size() const { return chunk_size; }
};

///! tag for a tiling loop
template <camp::idx_t default_chunk_size>
struct tile {
  camp::idx_t chunk_size;

  tile(camp::idx_t chunk_size_ = default_chunk_size) : chunk_size{chunk_size_}
  {
  }
  camp::idx_t get_chunk_size() const { return chunk_size; }
};

template <camp::idx_t Index, typename BaseWrapper>
struct TileWrapper : GenericWrapper<Index, BaseWrapper> {
  using Base = GenericWrapper<Index, BaseWrapper>;
  using Base::Base;
  template <typename InSegmentType>
  void operator()(InSegmentType s)
  {
    camp::get<Index>(Base::bw.data.st) = s;
    Base::bw();
  }
};

/**
 * @brief specialization of internal::thread_privatize for tile
 */
template <camp::idx_t Index, typename BW>
auto thread_privatize(const nested::TileWrapper<Index, BW> &item)
    -> NestedPrivatizer<nested::TileWrapper<Index, BW>>
{
  return NestedPrivatizer<nested::TileWrapper<Index, BW>>{item};
}

template <typename Iterable>
struct IterableTiler {
  using value_type = camp::decay<Iterable>;

  class iterator
  {
    const IterableTiler &itiler;
    const Index_type block_id;

  public:
    using value_type = camp::decay<Iterable>;
    using difference_type = camp::idx_t;
    using pointer = value_type *;
    using reference = value_type &;
    using iterator_category = std::random_access_iterator_tag;

    constexpr iterator(IterableTiler const &itiler_, Index_type block_id_)
        : itiler{itiler_}, block_id{block_id_}
    {
    }

    value_type operator*()
    {
      auto start = block_id * itiler.block_size;
      return itiler.it.slice(start, itiler.block_size);
    }

    inline difference_type operator-(const iterator &rhs) const
    {
      return static_cast<difference_type>(block_id)
             - static_cast<difference_type>(rhs.block_id);
    }

    inline iterator operator-(const difference_type &rhs) const
    {
      return iterator(itiler, block_id - rhs);
    }

    inline iterator operator+(const difference_type &rhs) const
    {
      return iterator(itiler,
                      block_id + rhs >= itiler.num_blocks ? itiler.num_blocks
                                                          : block_id + rhs);
    }

    inline value_type operator[](difference_type rhs) const
    {
      return *((*this) + rhs);
    }

    inline bool operator!=(const IterableTiler &rhs) const
    {
      return block_id != rhs.block_id;
    }

    inline bool operator<(const IterableTiler &rhs) const
    {
      return block_id < rhs.block_id;
    }
  };

  IterableTiler(const Iterable &it_, camp::idx_t block_size_)
      : it{it_}, block_size{block_size_}
  {
    using std::begin;
    using std::end;
    using std::distance;
    dist = distance(begin(it), end(it));
    num_blocks = dist / block_size;
    if (dist % block_size) num_blocks += 1;
  }

  iterator begin() { return iterator(*this, 0); }

  iterator end() { return iterator(*this, num_blocks); }

  value_type it;
  camp::idx_t block_size;
  camp::idx_t num_blocks;
  camp::idx_t dist;
};

template <typename TPol, typename EPol, camp::idx_t Index>
struct Executor<Tile<Index, TPol, EPol>> {
  using TileType = Tile<Index, TPol, EPol>;
  template <typename WrappedBody>
  void operator()(TileType const &fp, WrappedBody const &wrap)
  {
    auto const &st = camp::get<Index>(wrap.data.st);
    IterableTiler<decltype(st)> tiled_iterable(st, fp.tpol.get_chunk_size());
    impl::forall(fp.epol,
                 tiled_iterable,
                 TileWrapper<Index, WrappedBody>{wrap});
    // Set range back to original values
    camp::get<Index>(wrap.data.st) = tiled_iterable.it;
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

#endif /* RAJA_pattern_nested_HPP */
