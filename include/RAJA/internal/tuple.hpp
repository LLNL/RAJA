#ifndef RAJA_internal_tuple_HPP__
#define RAJA_internal_tuple_HPP__

/*!
 * \file
 *
 * \brief   Exceptionally basic tuple for host-device support
 */

#include <RAJA/internal/LegacyCompatibility.hpp>
#include <RAJA/util/defines.hpp>

#include "RAJA/external/metal.hpp"

#include <iostream>
#include <type_traits>

namespace RAJA
{
namespace util
{
template <metal::int_ I>
using make_nums =
    metal::iota<metal::number<0>, metal::number<I>, metal::number<1>>;

template <typename... Rest>
struct tuple;

namespace internal
{
template <class T>
struct unwrap_refwrapper {
  using type = T;
};

template <class T>
struct unwrap_refwrapper<std::reference_wrapper<T>> {
  using type = T&;
};

template <class T>
using special_decay_t =
    typename unwrap_refwrapper<typename std::decay<T>::type>::type;
}

template <typename... Args>
RAJA_HOST_DEVICE constexpr auto make_tuple(Args&&... args)
    -> tuple<internal::special_decay_t<Args>...>;

namespace internal
{
template <typename... Ts>
void ignore_args(Ts... args)
{
}
template <metal::int_ index, typename Type>
struct tuple_storage {
  tuple_storage() = default;
  RAJA_HOST_DEVICE constexpr tuple_storage(Type val) : val{val} {}

  RAJA_HOST_DEVICE
  constexpr const Type& get_inner() const noexcept { return val; }

  RAJA_CXX14_CONSTEXPR
  RAJA_HOST_DEVICE
  Type& get_inner() noexcept { return val; }

public:
  Type val;
};

template <typename... Types>
struct tuple_helper;

template <>
struct tuple_helper<metal::list<>, metal::list<>>{};

template <typename... Types, metal::int_... Indices>
struct tuple_helper<metal::list<metal::number<Indices>...>, metal::list<Types...>>
    : public internal::tuple_storage<Indices, Types>... {

  tuple_helper() = default;

  using Self = tuple_helper<metal::numbers<Indices...>, Types...>;
  RAJA_HOST_DEVICE constexpr tuple_helper(Types... args)
      : internal::tuple_storage<Indices, Types>(std::forward<Types>(args))...
  {
  }

  template <typename... RTypes>
  RAJA_HOST_DEVICE RAJA_CXX14_CONSTEXPR Self& operator=(
      const tuple_helper<metal::numbers<Indices...>, RTypes...>& rhs)
  {
    ignore_args((this->tuple_storage<Indices, Types>::get_inner() =
                     rhs.tuple_storage<Indices, RTypes>::get_inner())...);
    return *this;
  }
};
}

template <typename T, metal::int_ I>
using tpl_get_ret = metal::at<typename T::TList, metal::number<I>>;
template <typename T, metal::int_ I>
using tpl_get_store = internal::tuple_storage<I, tpl_get_ret<T, I>>;

template <typename... Elements>
struct tuple : public internal::tuple_helper<make_nums<sizeof...(Elements)>,
                                             metal::list<Elements...>> {
  using TList = metal::list<Elements...>;

private:
  using Self = tuple<Elements...>;
  using Base = internal::tuple_helper<make_nums<sizeof...(Elements)>,
                                      metal::list<Elements...>>;

public:
  // Constructors
  tuple() = default;
  tuple(tuple const&) = default;
  tuple(tuple&&) = default;
  tuple& operator=(tuple const& rhs) = default;
  tuple& operator=(tuple&& rhs) = default;

  template <typename... OtherTypes>
  RAJA_HOST_DEVICE constexpr explicit tuple(OtherTypes&&... rest)
      : Base{std::forward<OtherTypes>(rest)...}
  {
  }

  template <typename... RTypes>
  RAJA_HOST_DEVICE RAJA_CXX14_CONSTEXPR Self& operator=(
      const tuple<RTypes...>& rhs)
  {
    Base::operator=(rhs);
    return *this;
  }

  template <metal::int_ index>
  RAJA_HOST_DEVICE auto get() noexcept -> tpl_get_ret<Self, index>
  {
    static_assert(sizeof...(Elements) > index, "index out of range");
    return tpl_get_store<Self, index>::get_inner();
  }
  template <metal::int_ index>
  RAJA_HOST_DEVICE auto get() const noexcept -> tpl_get_ret<Self, index>
  {
    static_assert(sizeof...(Elements) > index, "index out of range");
    return tpl_get_store<Self, index>::get_inner();
  }
};

template <metal::int_ i, typename T>
struct tuple_element;
template <metal::int_ i, typename... Types>
struct tuple_element<i, tuple<Types...>> {
  using type = metal::at<metal::list<Types...>, metal::number<i>>;
};

template <int index, typename... Args>
RAJA_HOST_DEVICE constexpr auto get(const tuple<Args...>& t) noexcept
    -> tpl_get_ret<tuple<Args...>, index> const &
{
  static_assert(sizeof...(Args) > index, "index out of range");
  return t.tpl_get_store<tuple<Args...>, index>::get_inner();
}

template <int index, typename... Args>
RAJA_HOST_DEVICE constexpr auto get(tuple<Args...>& t) noexcept
    -> tpl_get_ret<tuple<Args...>, index> &
{
  static_assert(sizeof...(Args) > index, "index out of range");
  return t.tpl_get_store<tuple<Args...>, index>::get_inner();
}

template <typename Tuple>
struct tuple_size;

template <typename... Args>
struct tuple_size<tuple<Args...>> {
  static constexpr size_t value = sizeof...(Args);
};

template <typename... Args>
struct tuple_size<tuple<Args...>&> {
  static constexpr size_t value = sizeof...(Args);
};

template <typename... Args>
RAJA_HOST_DEVICE constexpr auto make_tuple(Args&&... args)
    -> tuple<internal::special_decay_t<Args>...>
{
  return tuple<internal::special_decay_t<Args>...>{std::forward<Args>(args)...};
}

template <typename... Args>
RAJA_HOST_DEVICE constexpr auto forward_as_tuple(Args&&... args) noexcept
    -> tuple<Args&&...>
{
  return tuple<Args&&...>(std::forward<Args>(args)...);
}

template <class... Types>
RAJA_HOST_DEVICE constexpr tuple<Types&...> tie(Types&... args) noexcept
{
  return tuple<Types&...>{args...};
}

template <typename... Lelem,
          typename... Relem,
          metal::int_... Lidx,
          metal::int_... Ridx>
RAJA_HOST_DEVICE constexpr auto tuple_cat_pair(tuple<Lelem...>&& l,
                                               metal::numbers<Lidx...>,
                                               tuple<Relem...>&& r,
                                               metal::numbers<Ridx...>) noexcept
    -> tuple<Lelem..., Relem...>
{
  return make_tuple(get<Lidx>(l)..., get<Ridx>(r)...);
}

template <typename Fn, metal::int_... Sequence, typename TupleLike>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto invoke_with_order(
    TupleLike&& t,
    Fn&& f,
    metal::numbers<Sequence...>) -> decltype(f(get<Sequence>(t)...))
{
  return f(get<Sequence>(t)...);
}

template <typename Fn, typename TupleLike>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto invoke(TupleLike&& t, Fn&& f)
    -> decltype(
        invoke_with_order(t,
                          f,
                          make_nums<tuple_size<TupleLike>::value>{}))
{
  return invoke_with_order(t,
                           f,
                           make_nums<tuple_size<TupleLike>::value>{});
}
}
}

namespace internal
{
template <class Tuple, metal::int_... Idxs>
void print_tuple(std::ostream& os, Tuple const& t, metal::list<metal::number<Idxs>...>)
{
  RAJA::util::internal::ignore_args(
      (void*)&(os << (Idxs == 0 ? "" : ", ") << RAJA::util::get<Idxs>(t))...);
}
}

template <class... Args>
auto operator<<(std::ostream& os, RAJA::util::tuple<Args...> const& t)
    -> std::ostream&
{
  os << "(";
  internal::print_tuple(os, t, RAJA::util::make_nums<sizeof...(Args)>{});
  return os << ")";
}


#endif /* RAJA_internal_tuple_HPP__ */
