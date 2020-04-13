/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for multi-iterator Zip Views.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#ifndef RAJA_util_zip_ref_HPP
#define RAJA_util_zip_ref_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "RAJA/pattern/detail/algorithm.hpp"
#include "RAJA/util/camp_aliases.hpp"
#include "RAJA/util/concepts.hpp"

namespace RAJA
{

namespace detail
{

// ignore arguments
template< typename ... Args >
RAJA_HOST_DEVICE inline void sink(Args&&...) { };

struct Noop
{
  template < typename T >
  RAJA_HOST_DEVICE RAJA_INLINE auto operator()(T&& t) const
    -> decltype(std::forward<T>(t))
  {
    return std::forward<T>(t);
  }
};

struct Move
{
  template < typename T >
  RAJA_HOST_DEVICE RAJA_INLINE auto operator()(T&& t) const
    -> decltype(std::move(t))
  {
    return std::move(t);
  }
};

struct PreInc
{
  template< typename Iter >
  RAJA_HOST_DEVICE inline auto operator()(Iter&& iter) const
    -> decltype(++std::forward<Iter>(iter))
  {
    return ++std::forward<Iter>(iter);
  }
};

struct PreDec
{
  template< typename Iter >
  RAJA_HOST_DEVICE inline auto operator()(Iter&& iter) const
    -> decltype(--std::forward<Iter>(iter))
  {
    return --std::forward<Iter>(iter);
  }
};

template < typename difference_type >
struct PlusEq
{
  const difference_type& rhs;
  template< typename Iter >
  RAJA_HOST_DEVICE inline auto operator()(Iter&& iter) const
    -> decltype(std::forward<Iter>(iter) += rhs)
  {
    return std::forward<Iter>(iter) += rhs;
  }
};

template < typename difference_type >
struct MinusEq
{
  const difference_type& rhs;
  template< typename Iter >
  RAJA_HOST_DEVICE inline auto operator()(Iter&& iter) const
    -> decltype(std::forward<Iter>(iter) -= rhs)
  {
    return std::forward<Iter>(iter) -= rhs;
  }
};

struct DeRef
{
  template< typename Iter >
  RAJA_HOST_DEVICE inline auto operator()(Iter&& iter) const
    -> decltype(*std::forward<Iter>(iter))
  {
    return *std::forward<Iter>(iter);
  }
};

struct Swap
{
  template< typename T0, typename T1 >
  RAJA_HOST_DEVICE inline int operator()(T0&& t0, T1&& t1) const
  {
    using camp::safe_swap;
    safe_swap(std::forward<T0>(t0), std::forward<T1>(t1));
    return 1;
  }
};

struct IterSwap
{
  template< typename T0, typename T1 >
  RAJA_HOST_DEVICE inline int operator()(T0&& t0, T1&& t1) const
  {
    using RAJA::safe_iter_swap;
    safe_iter_swap(std::forward<T0>(t0), std::forward<T1>(t1));
    return 1;
  }
};


template < typename Tuple, typename F, camp::idx_t... Is >
RAJA_HOST_DEVICE inline
void zip_for_each_impl(Tuple&& t, F&& f, camp::idx_seq<Is...>)
{
  RAJA::detail::sink(std::forward<F>(f)(std::forward<Tuple>(t).template get<Is>())...);
}

template < typename Tuple0, typename Tuple1, typename F, camp::idx_t... Is >
RAJA_HOST_DEVICE inline
void zip_for_each_impl(Tuple0&& t0, Tuple1&& t1, F&& f, camp::idx_seq<Is...>)
{
  RAJA::detail::sink(std::forward<F>(f)(std::forward<Tuple0>(t0).template get<Is>(), std::forward<Tuple1>(t1).template get<Is>())...);
}

template < typename Tuple, typename F >
RAJA_HOST_DEVICE inline
void zip_for_each(Tuple&& t, F&& f)
{
  zip_for_each_impl(std::forward<Tuple>(t), std::forward<F>(f), typename camp::decay<Tuple>::IdxSeq{});
}

template < typename Tuple0, typename Tuple1, typename F >
RAJA_HOST_DEVICE inline
void zip_for_each(Tuple0&& t0, Tuple1&& t1, F&& f)
{
  static_assert(std::is_same<typename camp::decay<Tuple0>::IdxSeq, typename camp::decay<Tuple1>::IdxSeq>::value,
      "Tuple0 and Tuple1 must have the same size");
  zip_for_each_impl(std::forward<Tuple0>(t0), std::forward<Tuple1>(t1), std::forward<F>(f), typename camp::decay<Tuple0>::IdxSeq{});
}


template < bool is_val, typename ... Ts >
struct zip_tuple
{
  using value_type = RAJA::tuple<Ts...>;

  template < typename ... Os >
  using opp_tuple = zip_tuple<!is_val, Os...>;

  using IdxSeq = camp::make_idx_seq_t<sizeof...(Ts)>;

  // constructor from values convertible to Ts
  template < typename ... Os
           , typename = concepts::enable_if<DefineConcept(concepts::convertible_to<Ts>(camp::val<Os&&>()))...> >
  RAJA_HOST_DEVICE RAJA_INLINE zip_tuple(Os&&... os)
    : m_tuple(std::forward<Os>(os)...) { }

  // assignment from values convertible to Ts
  // template < typename ... Os
  //          , typename = concepts::enable_if<DefineConcept(concepts::convertible_to<typename std::remove_reference<Ts>::type>(camp::val<Os&&>()))...> >
  // zip_tuple& operator=(Os&&... os) { RAJA::detail::sink(get<Is>() = std::forward<Os>(os)...); return *this; }

  // copy and move constructors
  RAJA_HOST_DEVICE RAJA_INLINE zip_tuple(zip_tuple &      o)
    : zip_tuple(          o , IdxSeq{}) { }
  RAJA_HOST_DEVICE RAJA_INLINE zip_tuple(zip_tuple const& o)
    : zip_tuple(          o , IdxSeq{}) { }
  RAJA_HOST_DEVICE RAJA_INLINE zip_tuple(zip_tuple &&     o)
    : zip_tuple(std::move(o), IdxSeq{}) { } // move if is_val, noop otherwise

  // copy and move assignment operators
  RAJA_HOST_DEVICE RAJA_INLINE zip_tuple& operator=(zip_tuple &      o)
    { if (this != &o) { assign_helper(          o , IdxSeq{}); } return *this; }
  RAJA_HOST_DEVICE RAJA_INLINE zip_tuple& operator=(zip_tuple const& o)
    { if (this != &o) { assign_helper(          o , IdxSeq{}); } return *this; }
  RAJA_HOST_DEVICE RAJA_INLINE zip_tuple& operator=(zip_tuple &&     o)
    { if (this != &o) { assign_helper(std::move(o), IdxSeq{}); } return *this; }

  // copy and move constructors from opp_tuple type zip_tuples
  template < typename ... Os, typename = typename std::enable_if<sizeof...(Ts) == sizeof...(Os)>::type >
  RAJA_HOST_DEVICE RAJA_INLINE zip_tuple(opp_tuple<Os...> &      o)
    : zip_tuple(          o , IdxSeq{}) { }
  template < typename ... Os, typename = typename std::enable_if<sizeof...(Ts) == sizeof...(Os)>::type >
  RAJA_HOST_DEVICE RAJA_INLINE zip_tuple(opp_tuple<Os...> const& o)
    : zip_tuple(          o , IdxSeq{}) { }
  template < typename ... Os, typename = typename std::enable_if<sizeof...(Ts) == sizeof...(Os)>::type >
  RAJA_HOST_DEVICE RAJA_INLINE zip_tuple(opp_tuple<Os...> &&     o)
    : zip_tuple(std::move(o), IdxSeq{}) { } // move if is_val, noop otherwise

  // copy and move assignment operators from opp_tuple type zip_tuples
  template < typename ... Os >
  RAJA_HOST_DEVICE RAJA_INLINE zip_tuple& operator=(opp_tuple<Os...> &      o)
  { assign_helper(          o , IdxSeq{}); return *this; }
  template < typename ... Os >
  RAJA_HOST_DEVICE RAJA_INLINE zip_tuple& operator=(opp_tuple<Os...> const& o)
  { assign_helper(          o , IdxSeq{}); return *this; }
  template < typename ... Os >
  RAJA_HOST_DEVICE RAJA_INLINE zip_tuple& operator=(opp_tuple<Os...> &&     o)
  { assign_helper(std::move(o), IdxSeq{}); return *this; }

  // get member functions for zip_tuples
  // the reference type returned by get depends on the reference type
  // of the zip_tuple that get is called on
  template < camp::idx_t I >
  RAJA_HOST_DEVICE RAJA_INLINE camp::tuple_element_t<I, value_type> & get() &
  { return RAJA::get<I>(m_tuple); }
  template < camp::idx_t I >
  RAJA_HOST_DEVICE RAJA_INLINE camp::tuple_element_t<I, value_type> const& get() const&
  { return RAJA::get<I>(m_tuple); }
  template < camp::idx_t I >
  RAJA_HOST_DEVICE RAJA_INLINE typename std::remove_reference<camp::tuple_element_t<I, value_type>>::type && get() &&
  { return std::move(RAJA::get<I>(m_tuple)); }

  // safe_swap that calls swap on each pair in the tuple
  RAJA_HOST_DEVICE friend RAJA_INLINE void safe_swap(zip_tuple& lhs, zip_tuple& rhs)
  {
    zip_for_each(lhs, rhs, detail::Swap{});
  }
  // safe_swap for swapping zip_tuples with opposite is_val, calls swap on each pair in the tuple
  template < typename ... Os,
         typename = typename std::enable_if<(sizeof...(Ts) == sizeof...(Os))>::type >
  RAJA_HOST_DEVICE friend RAJA_INLINE void safe_swap(zip_tuple& lhs, opp_tuple<Os...>& rhs)
  {
    zip_for_each(lhs, rhs, detail::Swap{});
  }

  // allow printing of zip_tuples by printing tuple
  friend inline std::ostream& operator<<(std::ostream& o, zip_tuple const& v)
  {
    return o << v.m_tuple;
  }

private:
  // move if is_val is true, otherwise copy in move constructor
  // this allows values to be moved, and references to stay lvalue references
  using ValMover = typename std::conditional<is_val, Move, Noop>::type;

  value_type m_tuple;

  // copy and move constructor helpers
  template < camp::idx_t ... Is >
  RAJA_HOST_DEVICE RAJA_INLINE zip_tuple(zip_tuple &      o, camp::idx_seq<Is...>)
    : zip_tuple(           o .template get<Is>()...) { }
  template < camp::idx_t ... Is >
  RAJA_HOST_DEVICE RAJA_INLINE zip_tuple(zip_tuple const& o, camp::idx_seq<Is...>)
    : zip_tuple(           o .template get<Is>()...) { }
  template < camp::idx_t ... Is >
  RAJA_HOST_DEVICE RAJA_INLINE zip_tuple(zip_tuple &&     o, camp::idx_seq<Is...>)
    : zip_tuple(ValMover{}(o).template get<Is>()...) { } // move if is_val, noop otherwise

  // copy and move assignment operator helpers
  template < camp::idx_t ... Is >
  RAJA_HOST_DEVICE RAJA_INLINE zip_tuple& assign_helper(zip_tuple &      o, camp::idx_seq<Is...>)
  { if (this != &o) { RAJA::detail::sink(get<Is>() =           o .template get<Is>()...); } return *this; }
  template < camp::idx_t ... Is >
  RAJA_HOST_DEVICE RAJA_INLINE zip_tuple& assign_helper(zip_tuple const& o, camp::idx_seq<Is...>)
  { if (this != &o) { RAJA::detail::sink(get<Is>() =           o .template get<Is>()...); } return *this; }
  template < camp::idx_t ... Is >
  RAJA_HOST_DEVICE RAJA_INLINE zip_tuple& assign_helper(zip_tuple &&     o, camp::idx_seq<Is...>)
  { if (this != &o) { RAJA::detail::sink(get<Is>() = std::move(o).template get<Is>()...); } return *this; }

  // copy and move constructor helpers from opp_tuple type zip_tuples
  template < typename ... Os, camp::idx_t ... Is >
  RAJA_HOST_DEVICE RAJA_INLINE zip_tuple(opp_tuple<Os...> &      o, camp::idx_seq<Is...>)
    : zip_tuple(           o .template get<Is>()...) { }
  template < typename ... Os, camp::idx_t ... Is >
  RAJA_HOST_DEVICE RAJA_INLINE zip_tuple(opp_tuple<Os...> const& o, camp::idx_seq<Is...>)
    : zip_tuple(           o .template get<Is>()...) { }
  template < typename ... Os, camp::idx_t ... Is >
  RAJA_HOST_DEVICE RAJA_INLINE zip_tuple(opp_tuple<Os...> &&     o, camp::idx_seq<Is...>)
    : zip_tuple(ValMover{}(o).template get<Is>()...) { } // move if is_val, noop otherwise

  // copy and move assignment operator helpers from opp_tuple type zip_tuples
  template < typename ... Os, camp::idx_t ... Is >
  RAJA_HOST_DEVICE RAJA_INLINE zip_tuple& assign_helper(opp_tuple<Os...> &      o, camp::idx_seq<Is...>)
  { RAJA::detail::sink(get<Is>() =           o .template get<Is>()...); return *this; }
  template < typename ... Os, camp::idx_t ... Is >
  RAJA_HOST_DEVICE RAJA_INLINE zip_tuple& assign_helper(opp_tuple<Os...> const& o, camp::idx_seq<Is...>)
  { RAJA::detail::sink(get<Is>() =           o .template get<Is>()...); return *this; }
  template < typename ... Os, camp::idx_t ... Is >
  RAJA_HOST_DEVICE RAJA_INLINE zip_tuple& assign_helper(opp_tuple<Os...> &&     o, camp::idx_seq<Is...>)
  { RAJA::detail::sink(get<Is>() = std::move(o).template get<Is>()...); return *this; }

};

// alias zip_ref to zip_tuple capable of storing references (!is_val)
template < typename ... Ts >
using zip_ref = zip_tuple<false, Ts...>;

// alias zip_val to zip_tuple suitable for storing values (is_val)
template < typename ... Ts >
using zip_val = zip_tuple<true, Ts...>;

}  // end namespace detail

}  // end namespace RAJA

#endif
