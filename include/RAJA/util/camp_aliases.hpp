/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file with aliases to camp types.
 *
 * The aliases included here are the ones that may be exposed through the
 * RAJA API based on our unit tests and examples. As you build new tests
 * and examples and you find that other camp types are exposed, please
 * add them to this file.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_CAMP_ALIASES_HPP
#define RAJA_CAMP_ALIASES_HPP

#include "RAJA/config.hpp"
#include "RAJA/util/macros.hpp"

#include "camp/defines.hpp"
#include "camp/list/list.hpp"
#include "camp/tuple.hpp"
#include "camp/resource.hpp"

namespace RAJA
{

using ::camp::at_v;

using ::camp::list;

using ::camp::idx_t;

using ::camp::make_tuple;

using ::camp::tuple;

using ::camp::resources::Platform;

// make own tuple_element
template < camp::idx_t I, typename Tuple >
struct tuple_element;

// specialization for RAJA/camp::tuple
template < camp::idx_t I, typename ... Ts >
struct tuple_element<I, tuple<Ts...>>
  : camp::tuple_element<I, tuple<Ts...>>
{ };

// convenience alias
template < camp::idx_t I, typename Tuple >
using tuple_element_t = typename tuple_element<I, Tuple>::type;

// get function overloads for tuple
// the reference type returned by get depends on the reference type
// of the zip_tuple that get is called on
template < camp::idx_t I, typename ... Ts >
// RAJA_HOST_DEVICE RAJA_INLINE                                RAJA::tuple_element_t<I, tuple<Ts...>>             &
// RAJA_HOST_DEVICE RAJA_INLINE decltype(camp::get<I>(camp::val<tuple<Ts...>      & >()))
RAJA_HOST_DEVICE RAJA_INLINE auto get(tuple<Ts...>      &  t)
  -> decltype(camp::get<I>(t))
{ return camp::get<I>(          t ); }
template < camp::idx_t I, typename ... Ts >
// RAJA_HOST_DEVICE RAJA_INLINE                                RAJA::tuple_element_t<I, tuple<Ts...>>        const&
// RAJA_HOST_DEVICE RAJA_INLINE decltype(camp::get<I>(camp::val<tuple<Ts...> const& >()))
RAJA_HOST_DEVICE RAJA_INLINE auto get(tuple<Ts...> const&  t)
  -> decltype(camp::get<I>(t))
{ return camp::get<I>(          t ); }
template < camp::idx_t I, typename ... Ts >
// RAJA_HOST_DEVICE RAJA_INLINE typename std::remove_reference<RAJA::tuple_element_t<I, tuple<Ts...>>>::type      &&
// RAJA_HOST_DEVICE RAJA_INLINE decltype(camp::get<I>(camp::val<tuple<Ts...>      &&>()))
RAJA_HOST_DEVICE RAJA_INLINE auto get(tuple<Ts...>      && t)
  -> decltype(camp::get<I>(std::move(t)))
{ return camp::get<I>(std::move(t)); }
template < camp::idx_t I, typename ... Ts >
// RAJA_HOST_DEVICE RAJA_INLINE typename std::remove_reference<RAJA::tuple_element_t<I, tuple<Ts...>>>::type const&&
// RAJA_HOST_DEVICE RAJA_INLINE decltype(camp::get<I>(camp::val<tuple<Ts...> const&&>()))
RAJA_HOST_DEVICE RAJA_INLINE auto get(tuple<Ts...> const&& t)
  -> decltype(camp::get<I>(std::move(t)))
{ return camp::get<I>(std::move(t)); }

}  // end namespace RAJA

#endif /* RAJA_CAMP_ALIASES_HPP */
