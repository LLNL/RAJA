#ifndef RAJA_pattern_nested_internal_HPP
#define RAJA_pattern_nested_internal_HPP

#include "RAJA/RAJA.hpp"
#include "RAJA/config.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/util/types.hpp"
#include "RAJA/policy/cuda.hpp"

#include "camp/camp.hpp"
#include "camp/concepts.hpp"
#include "camp/tuple.hpp"

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

}  // end namespace internal
}  // end namespace nested
}  // end namespace RAJA


#endif /* RAJA_pattern_nested_internal_HPP */
