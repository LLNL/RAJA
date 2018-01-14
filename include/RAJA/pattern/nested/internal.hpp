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

#ifndef RAJA_pattern_nested_internal_HPP
#define RAJA_pattern_nested_internal_HPP

#include "RAJA/config.hpp"
#include "RAJA/policy/cuda.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/util/types.hpp"

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
struct CollapseBase {
};
template <camp::idx_t ArgumentId, typename Policy>
struct ForTraitBase : public ForBase {
  constexpr static camp::idx_t index_val = ArgumentId;
  using index = camp::num<ArgumentId>;
  using index_type = camp::nil;  // default to invalid type
  using policy_type = Policy;
  using type = ForTraitBase;  // make camp::value compatible
};

using has_for_list = typename camp::bind_front<std::is_base_of, ForList>::type;

template <typename T>
using get_space_list = typename T::as_space_list;


/*
 * Get a camp::list of execution policies from a RAJA::nested::Policy to be
 * used to determine the execution space.
 *
 * This extracts the execution policy from each For<>, and extracts the
 * COLLAPSE policy P from inside of Collapse<P, ...>.  It drops all of the
 * For<>'s inside of a collapse.
 *
 */
template <typename Seq>
using get_space_policies = typename camp::flatten<
    typename camp::transform<get_space_list,
                             typename camp::filter_l<has_for_list,
                                                     Seq>::type>::type>::type;



template <typename Iterator>
struct iterable_value_type_getter {
  using type = typename Iterator::iterator::value_type;
};

template <typename Segments>
using value_type_list_from_segments =
    typename camp::transform<iterable_value_type_getter, Segments>::type;


template <typename Segments>
using index_tuple_from_segments =
    typename camp::apply_l<camp::lambda<camp::tuple>,
                           value_type_list_from_segments<Segments> >::type;

}  // end namespace internal
}  // end namespace nested
}  // end namespace RAJA


#endif /* RAJA_pattern_nested_internal_HPP */
