#ifndef camp_tuple_HPP__
#define camp_tuple_HPP__

/*!
 * \file
 *
 * \brief   Exceptionally basic tuple for host-device support
 */

#include "camp/camp.hpp"

#include <iostream>
#include <type_traits>

namespace camp
{

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
CAMP_HOST_DEVICE constexpr auto make_tuple(Args&&... args)
    -> tuple<internal::special_decay_t<Args>...>;

namespace internal
{
  template <camp::idx_t index, typename Type>
  struct tuple_storage {
    tuple_storage() = default;
    constexpr tuple_storage(Type val) : val{val} {}

    constexpr const Type& get_inner() const noexcept { return val; }

    CAMP_CONSTEXPR14
    CAMP_HOST_DEVICE
    Type& get_inner() noexcept { return val; }

  public:
    Type val;
  };

  template <typename Indices, typename Typelist>
  struct tuple_helper;

  template <>
  struct tuple_helper<camp::idx_seq<>, camp::list<>> {
  };

  template <typename... Types, camp::idx_t... Indices>
  struct tuple_helper<camp::idx_seq<Indices...>, camp::list<Types...>>
      : public internal::tuple_storage<Indices, Types>... {

    tuple_helper() = default;

    CAMP_HOST_DEVICE constexpr tuple_helper(Types... args)
        : internal::tuple_storage<Indices, Types>(std::forward<Types>(args))...
    {
    }

    template <typename... RTypes>
    CAMP_HOST_DEVICE tuple_helper& operator=(
        const tuple_helper<camp::idx_seq<Indices...>, RTypes...>& rhs)
    {
      return (camp::sink(
                  (this->tuple_storage<Indices, Types>::get_inner() =
                       rhs.tuple_storage<Indices, RTypes>::get_inner())...),
              *this);
    }
  };
}

template <typename T, camp::idx_t I>
using tpl_get_ret = camp::at_t<typename T::TList, I>;
template <typename T, camp::idx_t I>
using tpl_get_store = internal::tuple_storage<I, tpl_get_ret<T, I>>;

template <typename... Elements>
struct tuple
    : public internal::tuple_helper<camp::make_idx_seq_t<sizeof...(Elements)>,
                                    camp::list<Elements...>> {
  using TList = camp::list<Elements...>;

private:
  using Self = tuple<Elements...>;
  using Base = internal::tuple_helper<camp::make_idx_seq_t<sizeof...(Elements)>,
                                      camp::list<Elements...>>;

public:
  // Constructors
  tuple() = default;
  tuple(tuple const&) = default;
  tuple(tuple&&) = default;
  tuple& operator=(tuple const& rhs) = default;
  tuple& operator=(tuple&& rhs) = default;

  template <typename... OtherTypes>
  CAMP_HOST_DEVICE constexpr explicit tuple(OtherTypes&&... rest)
      : Base{std::forward<OtherTypes>(rest)...}
  {
  }

  template <typename... RTypes>
  CAMP_HOST_DEVICE CAMP_CONSTEXPR14 Self& operator=(const tuple<RTypes...>& rhs)
  {
    Base::operator=(rhs);
    return *this;
  }

  template <camp::idx_t index>
  CAMP_HOST_DEVICE auto get() noexcept -> tpl_get_ret<Self, index>&
  {
    static_assert(sizeof...(Elements) > index, "index out of range");
    return tpl_get_store<Self, index>::get_inner();
  }
  template <camp::idx_t index>
  CAMP_HOST_DEVICE auto get() const noexcept -> const tpl_get_ret<Self, index>&
  {
    static_assert(sizeof...(Elements) > index, "index out of range");
    return tpl_get_store<Self, index>::get_inner();
  }
};

template <camp::idx_t i, typename T>
struct tuple_element;
template <camp::idx_t i, typename... Types>
struct tuple_element<i, tuple<Types...>> {
  using type = camp::at_t<typename tuple<Types...>::TList, i>;
};
template <camp::idx_t i, typename T>
using tuple_element_t = typename tuple_element<i, T>::type;

template <int index, typename... Args>
CAMP_HOST_DEVICE constexpr auto get(const tuple<Args...>& t) noexcept
    -> tpl_get_ret<tuple<Args...>, index> const&
{
  static_assert(sizeof...(Args) > index, "index out of range");
  return t.tpl_get_store<tuple<Args...>, index>::get_inner();
}

template <int index, typename... Args>
CAMP_HOST_DEVICE constexpr auto get(tuple<Args...>& t) noexcept
    -> tpl_get_ret<tuple<Args...>, index>&
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
CAMP_HOST_DEVICE constexpr auto make_tuple(Args&&... args)
    -> tuple<internal::special_decay_t<Args>...>
{
  return tuple<internal::special_decay_t<Args>...>{std::forward<Args>(args)...};
}

template <typename... Args>
CAMP_HOST_DEVICE constexpr auto forward_as_tuple(Args&&... args) noexcept
    -> tuple<Args&&...>
{
  return tuple<Args&&...>(std::forward<Args>(args)...);
}

template <class... Types>
CAMP_HOST_DEVICE constexpr tuple<Types&...> tie(Types&... args) noexcept
{
  return tuple<Types&...>{args...};
}

template <typename... Lelem,
          typename... Relem,
          camp::idx_t... Lidx,
          camp::idx_t... Ridx>
CAMP_HOST_DEVICE constexpr auto tuple_cat_pair(tuple<Lelem...>&& l,
                                               camp::idx_seq<Lidx...>,
                                               tuple<Relem...>&& r,
                                               camp::idx_seq<Ridx...>) noexcept
    -> tuple<Lelem..., Relem...>
{
  return make_tuple(get<Lidx>(l)..., get<Ridx>(r)...);
}

template <typename Fn, camp::idx_t... Sequence, typename TupleLike>
CAMP_HOST_DEVICE constexpr auto invoke_with_order(TupleLike&& t,
                                                  Fn&& f,
                                                  camp::idx_seq<Sequence...>)
    -> decltype(f(get<Sequence>(t)...))
{
  return f(get<Sequence>(t)...);
}

template <typename Fn, typename TupleLike>
CAMP_HOST_DEVICE constexpr auto invoke(TupleLike&& t, Fn&& f) -> decltype(
    invoke_with_order(t,
                      f,
                      camp::make_idx_seq_t<tuple_size<TupleLike>::value>{}))
{
  return invoke_with_order(
      t, f, camp::make_idx_seq_t<tuple_size<TupleLike>::value>{});
}
}

namespace internal
{
template <class Tuple, camp::idx_t... Idxs>
void print_tuple(std::ostream& os, Tuple const& t, camp::idx_seq<Idxs...>)
{
  camp::sink((void*)&(os << (Idxs == 0 ? "" : ", ") << camp::get<Idxs>(t))...);
}
}

template <class... Args>
auto operator<<(std::ostream& os, camp::tuple<Args...> const& t)
    -> std::ostream&
{
  os << "(";
  internal::print_tuple(os, t, camp::make_idx_seq_t<sizeof...(Args)>{});
  return os << ")";
}


#endif /* camp_tuple_HPP__ */
