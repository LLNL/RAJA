/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for enable_if helpers.
 *
 *          These type functions are used heavily by the atomic operators.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_util_EnableIf_HPP
#define RAJA_util_EnableIf_HPP

#include "RAJA/config.hpp"

#include <type_traits>

#include "camp/list.hpp"
#include "camp/type_traits.hpp"

#include "RAJA/util/concepts.hpp"


namespace RAJA
{
namespace util
{


template <typename T, typename TypeList>
struct is_any_of;

template <typename T, typename... Types>
struct is_any_of<T, ::camp::list<Types...>>
    : ::RAJA::concepts::any_of<::camp::is_same<T, Types>...>
{};

template <typename T, typename TypeList>
using enable_if_is_any_of = std::enable_if_t<is_any_of<T, TypeList>::value, T>;

template <typename T, typename TypeList>
using enable_if_is_none_of =
    std::enable_if_t<::RAJA::concepts::negate<is_any_of<T, TypeList>>::value,
                     T>;


}  // namespace util
}  // namespace RAJA

#endif  // closing endif for header file include guard
