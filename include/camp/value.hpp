//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __CAMP_value_hpp
#define __CAMP_value_hpp

#include "camp/number/number.hpp"

namespace camp
{

/// \cond
namespace detail
{
  struct nothing;
}
/// \endcond

// TODO: document
template <typename val = detail::nothing>
struct value;

template <typename val>
struct value {
  using type = val;
};

template <>
struct value<detail::nothing> {
  using type = value;
};

/// A non-value, in truth tests evaluates to false
using nil = value<>;

/// Test whether a type is a valid camp value
template <typename Val>
struct is_value_s {
  using type = camp::t;
};

/// Test whether a type is a valid camp value
template <typename Val>
using is_value = typename is_value_s<Val>::type;
}  // namespace camp

#endif /* __CAMP_value_hpp */
