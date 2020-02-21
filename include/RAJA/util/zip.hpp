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


#ifndef RAJA_util_zip_HPP
#define RAJA_util_zip_HPP

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

  struct PreInc
  {
    template< typename Iter >
    RAJA_HOST_DEVICE inline auto operator()(Iter&& iter)
      -> decltype(++std::forward<Iter>(iter))
    {
      return ++std::forward<Iter>(iter);
    }
  };

  struct PreDec
  {
    template< typename Iter >
    RAJA_HOST_DEVICE inline auto operator()(Iter&& iter)
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
      using RAJA::swap;
      swap(std::forward<T0>(t0), std::forward<T1>(t1));
      return 1;
    }
  };

  struct IterSwap
  {
    template< typename T0, typename T1 >
    RAJA_HOST_DEVICE inline int operator()(T0&& t0, T1&& t1) const
    {
      using RAJA::iter_swap;
      iter_swap(std::forward<T0>(t0), std::forward<T1>(t1));
      return 1;
    }
  };


  template < typename... Ts >
  struct zip_ref;

  template < typename... Args >
  RAJA_HOST_DEVICE auto forward_as_zip_ref(Args&&... args)
    -> zip_ref<Args&&...>
  {
    return zip_ref<Args&&...>(std::forward<Args>(args)...);
  }

  template < typename Tuple, typename F, camp::idx_t... Is >
  RAJA_HOST_DEVICE inline
  auto zip_for_each_impl(Tuple&& t, F&& f, camp::idx_seq<Is...>)
    -> decltype(forward_as_zip_ref(std::forward<F>(f)(camp::get<Is>(std::forward<Tuple>(t)))...))
  {
    return forward_as_zip_ref(std::forward<F>(f)(camp::get<Is>(std::forward<Tuple>(t)))...);
  }

  template < typename Tuple0, typename Tuple1, typename F, camp::idx_t... Is >
  RAJA_HOST_DEVICE inline
  auto zip_for_each_impl(Tuple0&& t0, Tuple1&& t1, F&& f, camp::idx_seq<Is...>)
    -> decltype(forward_as_zip_ref(std::forward<F>(f)(camp::get<Is>(std::forward<Tuple0>(t0)), camp::get<Is>(std::forward<Tuple1>(t1)))...))
  {
    return forward_as_zip_ref(std::forward<F>(f)(camp::get<Is>(std::forward<Tuple0>(t0)), camp::get<Is>(std::forward<Tuple1>(t1)))...);
  }

  template < typename Tuple, typename F >
  RAJA_HOST_DEVICE inline
  auto zip_for_each(Tuple&& t, F&& f)
    -> decltype(zip_for_each_impl(std::forward<Tuple>(t), std::forward<F>(f), camp::make_idx_seq_t<camp::tuple_size<camp::decay<Tuple>>::value>{}))
  {
    return zip_for_each_impl(std::forward<Tuple>(t), std::forward<F>(f), camp::make_idx_seq_t<camp::tuple_size<camp::decay<Tuple>>::value>{});
  }

  template < typename Tuple0, typename Tuple1, typename F >
  RAJA_HOST_DEVICE inline
  auto zip_for_each(Tuple0&& t0, Tuple1&& t1, F&& f)
    -> decltype(zip_for_each_impl(std::forward<Tuple0>(t0), std::forward<Tuple1>(t1), std::forward<F>(f), camp::make_idx_seq_t<camp::tuple_size<camp::decay<Tuple0>>::value>{}))
  {
    static_assert(camp::tuple_size<camp::decay<Tuple0>>::value == camp::tuple_size<camp::decay<Tuple1>>::value,
        "Tuple0 and Tuple1 must have the same size");
    return zip_for_each_impl(std::forward<Tuple0>(t0), std::forward<Tuple1>(t1), std::forward<F>(f), camp::make_idx_seq_t<camp::tuple_size<camp::decay<Tuple0>>::value>{});
  }

  template < typename... Ts >
  struct zip_ref : RAJA::tuple<Ts...>
  {
    using base = RAJA::tuple<Ts...>;

    using base::base;

    template < typename... Os, camp::idx_t... Is >
    zip_ref(zip_ref<Os...> const& o, camp::idx_seq<Is...>)
      : base(camp::get<Is>(o)...)
    { }

    template < typename... Os >
    zip_ref(zip_ref<Os...> const& o)
      : zip_ref(o, camp::make_idx_seq_t<camp::tuple_size<typename zip_ref<Os...>::base>::value>{})
    { }

    template < size_t I >
    RAJA_HOST_DEVICE inline typename camp::tuple_element<I, base>::type& get()
    {
      return camp::get<I>((base&)*this);
    }

    template < size_t I >
    RAJA_HOST_DEVICE inline typename camp::tuple_element<I, base>::type const& get() const
    {
      return camp::get<I>((base const&)*this);
    }

    RAJA_HOST_DEVICE inline void swap(zip_ref& rhs)
    {
      zip_for_each((base&)(*this), (base&)rhs, detail::Swap{});
    }

    RAJA_HOST_DEVICE friend inline void swap(zip_ref& lhs, zip_ref& rhs)
    {
      lhs.swap(rhs);
    }
  };

} // namespace detail

template < typename... Iters >
struct ZipIterator
{
  static_assert(concepts::all_of<type_traits::is_random_access_iterator<Iters>...>::value,
      "ZipIterator can only contain random access iterators");
  static_assert(sizeof...(Iters) > 1,
      "ZipIterator must contain one or more iterators");

  using value_type = detail::zip_ref<typename std::iterator_traits<Iters>::value_type...>;
  using difference_type = std::ptrdiff_t;
  using pointer = void;
  using reference = detail::zip_ref<typename std::iterator_traits<Iters>::reference...>;
  using creference = detail::zip_ref<const typename std::iterator_traits<Iters>::reference...>;
  using iterator_category = std::random_access_iterator_tag;

  RAJA_HOST_DEVICE inline ZipIterator()
    : m_iterators()
  {
  }

  template < typename... Args, typename = concepts::enable_if<DefineConcept(concepts::convertible_to<Iters>(camp::val<Args>()))...> >
  RAJA_HOST_DEVICE inline ZipIterator(Args&&... args)
    : m_iterators(std::forward<Args>(args)...)
  {
  }

  RAJA_HOST_DEVICE inline ZipIterator(const ZipIterator& rhs)
    : m_iterators(rhs.m_iterators)
  {
  }
  RAJA_HOST_DEVICE inline ZipIterator(ZipIterator&& rhs)
    : m_iterators(std::move(rhs.m_iterators))
  {
  }

  RAJA_HOST_DEVICE inline ZipIterator& operator=(const ZipIterator& rhs)
  {
    m_iterators = rhs.m_iterators;
    return *this;
  }
  RAJA_HOST_DEVICE inline ZipIterator& operator=(ZipIterator&& rhs)
  {
    m_iterators = std::move(rhs.m_iterators);
    return *this;
  }


  RAJA_HOST_DEVICE inline difference_type get_stride() const { return 1; }

  RAJA_HOST_DEVICE inline bool operator==(const ZipIterator& rhs) const
  {
    return RAJA::get<0>(m_iterators) == RAJA::get<0>(rhs.m_iterators);
  }
  RAJA_HOST_DEVICE inline bool operator!=(const ZipIterator& rhs) const
  {
    return RAJA::get<0>(m_iterators) != RAJA::get<0>(rhs.m_iterators);
  }
  RAJA_HOST_DEVICE inline bool operator>(const ZipIterator& rhs) const
  {
    return RAJA::get<0>(m_iterators) > RAJA::get<0>(rhs.m_iterators);
  }
  RAJA_HOST_DEVICE inline bool operator<(const ZipIterator& rhs) const
  {
    return RAJA::get<0>(m_iterators) < RAJA::get<0>(rhs.m_iterators);
  }
  RAJA_HOST_DEVICE inline bool operator>=(const ZipIterator& rhs) const
  {
    return RAJA::get<0>(m_iterators) >= RAJA::get<0>(rhs.m_iterators);
  }
  RAJA_HOST_DEVICE inline bool operator<=(const ZipIterator& rhs) const
  {
    return RAJA::get<0>(m_iterators) <= RAJA::get<0>(rhs.m_iterators);
  }

  RAJA_HOST_DEVICE inline ZipIterator& operator++()
  {
    detail::zip_for_each(m_iterators, detail::PreInc{});
    return *this;
  }
  RAJA_HOST_DEVICE inline ZipIterator& operator--()
  {
    detail::zip_for_each(m_iterators, detail::PreDec{});
    return *this;
  }
  RAJA_HOST_DEVICE inline ZipIterator operator++(int)
  {
    ZipIterator tmp(*this);
    ++(*this);
    return tmp;
  }
  RAJA_HOST_DEVICE inline ZipIterator operator--(int)
  {
    ZipIterator tmp(*this);
    --(*this);
    return tmp;
  }

  RAJA_HOST_DEVICE inline ZipIterator& operator+=(
      const difference_type& rhs)
  {
    detail::zip_for_each(m_iterators, detail::PlusEq<difference_type>{rhs});
    return *this;
  }
  RAJA_HOST_DEVICE inline ZipIterator& operator-=(
      const difference_type& rhs)
  {
    detail::zip_for_each(m_iterators, detail::MinusEq<difference_type>{rhs});
    return *this;
  }

  RAJA_HOST_DEVICE inline difference_type operator-(
      const ZipIterator& rhs) const
  {
    return RAJA::get<0>(m_iterators) - RAJA::get<0>(rhs.m_iterators);
  }
  RAJA_HOST_DEVICE inline ZipIterator operator+(
      const difference_type& rhs) const
  {
    ZipIterator tmp(*this);
    tmp += rhs;
    return tmp;
  }
  RAJA_HOST_DEVICE inline ZipIterator operator-(
      const difference_type& rhs) const
  {
    ZipIterator tmp(*this);
    tmp -= rhs;
    return tmp;
  }
  RAJA_HOST_DEVICE friend constexpr ZipIterator operator+(
      difference_type lhs,
      const ZipIterator& rhs)
  {
    ZipIterator tmp(rhs);
    tmp += lhs;
    return tmp;
  }

  RAJA_HOST_DEVICE inline reference operator*() const
  {
    return detail::zip_for_each(m_iterators, detail::DeRef{});
  }
  // TODO:: figure out what to do with this
  RAJA_HOST_DEVICE inline reference operator->() const
  {
    return *(*this);
  }
  RAJA_HOST_DEVICE constexpr reference operator[](difference_type rhs) const
  {
    return *((*this) + rhs);
  }


  RAJA_HOST_DEVICE inline void iter_swap(ZipIterator& rhs)
  {
    zip_for_each(m_iterators, rhs.m_iterators, detail::IterSwap{});
  }

  RAJA_HOST_DEVICE friend inline void iter_swap(ZipIterator& lhs, ZipIterator& rhs)
  {
    lhs.iter_swap(rhs);
  }

private:
  RAJA::tuple<camp::decay<Iters>...> m_iterators;
};


template < typename... Args >
RAJA_HOST_DEVICE
auto zip(Args&&... args)
  -> ZipIterator<camp::decay<Args>...>
{
  return {std::forward<Args>(args)...};
}

}  // end namespace RAJA

#endif
