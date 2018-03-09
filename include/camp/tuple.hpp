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

template <typename TagList, typename... Elements>
class tagged_tuple;

template <typename Tuple>
struct tuple_size;

template <camp::idx_t i, typename T>
struct tuple_element {
  using type = camp::at_v<typename T::TList, i>;
};

template <camp::idx_t i, typename T>
using tuple_element_t = typename tuple_element<i, T>::type;

template <typename T, typename Tuple>
using tuple_ebt_t =
    typename tuple_element<camp::at_key<typename Tuple::TMap, T>::value,
                           Tuple>::type;


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
}  // namespace internal

template <typename... Args>
CAMP_HOST_DEVICE constexpr auto make_tuple(Args&&... args)
    -> tuple<internal::special_decay_t<Args>...>;

template <camp::idx_t index, class Tuple>
CAMP_HOST_DEVICE constexpr auto get(const Tuple& t) noexcept
    -> tuple_element_t<index, Tuple> const&;
template <camp::idx_t index, class Tuple>
CAMP_HOST_DEVICE constexpr auto get(Tuple& t) noexcept
    -> tuple_element_t<index, Tuple>&;

template <typename T, class Tuple>
CAMP_HOST_DEVICE constexpr auto get(const Tuple& t) noexcept
    -> tuple_ebt_t<T, Tuple> const&;
template <typename T, class Tuple>
CAMP_HOST_DEVICE constexpr auto get(Tuple& t) noexcept
    -> tuple_ebt_t<T, Tuple>&;

namespace internal
{
  template <camp::idx_t index, typename Type>
  struct tuple_storage {
    CAMP_HOST_DEVICE constexpr tuple_storage() : val(){};

    CAMP_SUPPRESS_HD_WARN
    CAMP_HOST_DEVICE constexpr tuple_storage(Type const& v) : val{v} {}

    CAMP_SUPPRESS_HD_WARN
    CAMP_HOST_DEVICE constexpr tuple_storage(Type&& v)
        : val{std::move(static_cast<Type>(v))}
    {
    }

    CAMP_HOST_DEVICE constexpr const Type& get_inner() const noexcept
    {
      return val;
    }

    CAMP_HOST_DEVICE CAMP_CONSTEXPR14 Type& get_inner() noexcept { return val; }

  public:
    Type val;
  };

  template <typename Indices, typename Typelist>
  struct tuple_helper;

  template <typename... Types, camp::idx_t... Indices>
  struct tuple_helper<camp::idx_seq<Indices...>, camp::list<Types...>>
      : public internal::tuple_storage<Indices, Types>... {
    CAMP_HOST_DEVICE constexpr tuple_helper() {}

    CAMP_HOST_DEVICE constexpr tuple_helper(Types const&... args)
        : internal::tuple_storage<Indices, Types>(args)...
    {
    }

    CAMP_HOST_DEVICE constexpr tuple_helper(const tuple_helper& rhs)
        : tuple_storage<Indices, Types>(
              rhs.tuple_storage<Indices, Types>::get_inner())...
    {
    }


    template <typename RTuple>
    CAMP_HOST_DEVICE tuple_helper& operator=(const RTuple& rhs)
    {
      return (camp::sink((this->tuple_storage<Indices, Types>::get_inner() =
                              get<Indices>(rhs))...),
              *this);
    }
  };

  template <typename Types, typename Indices>
  struct tag_map;
  template <typename... Types, camp::idx_t... Indices>
  struct tag_map<camp::list<Types...>, camp::idx_seq<Indices...>> {
    using type = camp::list<camp::list<Types, camp::num<Indices>>...>;
  };

  template <typename T, camp::idx_t I>
  using tpl_get_store = internal::tuple_storage<I, tuple_element_t<I, T>>;

}  // namespace internal


template <typename... Elements>
struct tuple {
private:
  using Self = tuple;
  using Base = internal::tuple_helper<camp::make_idx_seq_t<sizeof...(Elements)>,
                                      camp::list<Elements...>>;

public:
  using TList = camp::list<Elements...>;
  using TMap = typename internal::tag_map<
      camp::list<Elements...>,
      camp::make_idx_seq_t<sizeof...(Elements)>>::type;
  using type = tuple;

private:
  Base base;

  template <camp::idx_t index, class Tuple>
  CAMP_HOST_DEVICE constexpr friend auto get(const Tuple& t) noexcept
      -> tuple_element_t<index, Tuple> const&;
  template <camp::idx_t index, class Tuple>
  CAMP_HOST_DEVICE constexpr friend auto get(Tuple& t) noexcept
      -> tuple_element_t<index, Tuple>&;

  template <typename T, class Tuple>
  CAMP_HOST_DEVICE constexpr friend auto get(const Tuple& t) noexcept
      -> tuple_ebt_t<T, Tuple> const&;
  template <typename T, class Tuple>
  CAMP_HOST_DEVICE constexpr friend auto get(Tuple& t) noexcept
      -> tuple_ebt_t<T, Tuple>&;

public:
  // NOTE: __host__ __device__ on constructors causes warnings, and nothing else
  // Constructors
  CAMP_HOST_DEVICE constexpr tuple() : base() {}
  CAMP_HOST_DEVICE constexpr tuple(tuple const& o) : base(o.base) {}

  CAMP_HOST_DEVICE constexpr tuple(tuple&& o) : base(std::move(o.base)) {}

  CAMP_HOST_DEVICE tuple& operator=(tuple const& rhs) { base = rhs.base; return *this; }
  CAMP_HOST_DEVICE tuple& operator=(tuple&& rhs) { base = std::move(rhs.base); return *this; }

  CAMP_HOST_DEVICE constexpr explicit tuple(Elements const&... rest)
      : base{rest...}
  {
  }

  template <typename... RTypes>
  CAMP_HOST_DEVICE CAMP_CONSTEXPR14 Self& operator=(const tuple<RTypes...>& rhs)
  {
    base.operator=(rhs);
    return *this;
  }
};

// NOTE: this class should be built on top of tuple.  Any attempt to do that
// causes nvcc9.1 to die in EDG. As soon as nvcc9.1 goes away, this should be
// reduced to just a public derivation of tuple that overrides TMap.
template <typename TagList, typename... Elements>
class tagged_tuple : public tuple<Elements...>
{
  using Self = tagged_tuple;
  using Base = internal::tuple_helper<camp::make_idx_seq_t<sizeof...(Elements)>,
                                      camp::list<Elements...>>;

public:
  using TList = camp::list<Elements...>;
  using TMap = typename internal::
      tag_map<TagList, camp::make_idx_seq_t<sizeof...(Elements)>>::type;
  using type = tagged_tuple;

private:
  Base base;

  template <camp::idx_t index, class Tuple>
  CAMP_HOST_DEVICE constexpr friend auto get(const Tuple& t) noexcept
      -> tuple_element_t<index, Tuple> const&;
  template <camp::idx_t index, class Tuple>
  CAMP_HOST_DEVICE constexpr friend auto get(Tuple& t) noexcept
      -> tuple_element_t<index, Tuple>&;

  template <typename T, class Tuple>
  CAMP_HOST_DEVICE constexpr friend auto get(const Tuple& t) noexcept
      -> tuple_ebt_t<T, Tuple> const&;
  template <typename T, class Tuple>
  CAMP_HOST_DEVICE constexpr friend auto get(Tuple& t) noexcept
      -> tuple_ebt_t<T, Tuple>&;

public:
  // Constructors


public:
  // NOTE: __host__ __device__ on constructors causes warnings, and nothing else
  // Constructors
  CAMP_HOST_DEVICE constexpr tagged_tuple() : base() {}
  CAMP_HOST_DEVICE constexpr tagged_tuple(tagged_tuple const& o) : base(o.base) {}

  CAMP_HOST_DEVICE constexpr tagged_tuple(tagged_tuple&& o) : base(std::move(o.base)) {}

  CAMP_HOST_DEVICE tagged_tuple& operator=(tagged_tuple const& rhs) { base = rhs.base; return *this;}
  CAMP_HOST_DEVICE tagged_tuple& operator=(tagged_tuple&& rhs) { base = std::move(rhs.base); return *this;}

  CAMP_HOST_DEVICE constexpr explicit tagged_tuple(Elements const&... rest)
      : base{rest...}
  {
  }

  template <template<typename...> class T, typename... RTypes>
  CAMP_HOST_DEVICE CAMP_CONSTEXPR14 Self& operator=(const T<RTypes...>& rhs)
  {
    base.operator=(rhs);
    return *this;
  }
};

template <>
class tuple<>
{
public:
  using TList = camp::list<>;
  using TMap = TList;
  using type = tuple;
};

template <typename... Tags, typename... Args>
struct as_list_s<tagged_tuple<camp::list<Tags...>, Args...>> {
  using type = list<Args...>;
};

// by index
template <camp::idx_t index, class Tuple>
CAMP_HOST_DEVICE constexpr auto get(const Tuple& t) noexcept
    -> tuple_element_t<index, Tuple> const&
{
  using internal::tpl_get_store;
  static_assert(tuple_size<Tuple>::value > index, "index out of range");
  return static_cast<tpl_get_store<Tuple, index> const &>(t.base).get_inner();
}

template <camp::idx_t index, class Tuple>
CAMP_HOST_DEVICE constexpr auto get(Tuple& t) noexcept
    -> tuple_element_t<index, Tuple>&
{
  using internal::tpl_get_store;
  static_assert(tuple_size<Tuple>::value > index, "index out of range");
  return static_cast<tpl_get_store<Tuple, index>&>(t.base).get_inner();
}

// by type
template <typename T, class Tuple>
CAMP_HOST_DEVICE constexpr auto get(const Tuple& t) noexcept
    -> tuple_ebt_t<T, Tuple> const&
{
  using internal::tpl_get_store;
  using index_type = camp::at_key<typename Tuple::TMap, T>;
  static_assert(!std::is_same<camp::nil, index_type>::value,
                "invalid type index");

  return static_cast<tpl_get_store<Tuple, index_type::value>&>(t.base).get_inner();
}

template <typename T, class Tuple>
CAMP_HOST_DEVICE constexpr auto get(Tuple& t) noexcept -> tuple_ebt_t<T, Tuple>&
{
  using internal::tpl_get_store;
  using index_type = camp::at_key<typename Tuple::TMap, T>;
  static_assert(!std::is_same<camp::nil, index_type>::value,
                "invalid type index");

  return static_cast<tpl_get_store<Tuple, index_type::value>&>(t.base).get_inner();
}

template <typename... Args>
struct tuple_size<tuple<Args...>> {
  static constexpr size_t value = sizeof...(Args);
};

template <typename... Args>
struct tuple_size<tuple<Args...>&> {
  static constexpr size_t value = sizeof...(Args);
};

template <typename L, typename... Args>
struct tuple_size<tagged_tuple<L, Args...>> {
  static constexpr size_t value = sizeof...(Args);
};

template <typename L, typename... Args>
struct tuple_size<tagged_tuple<L, Args...>&> {
  static constexpr size_t value = sizeof...(Args);
};

template <typename... Args>
CAMP_HOST_DEVICE constexpr auto make_tuple(Args&&... args)
    -> tuple<internal::special_decay_t<Args>...>
{
  return tuple<internal::special_decay_t<Args>...>{std::forward<Args>(args)...};
}

template <typename TagList, typename... Args>
CAMP_HOST_DEVICE constexpr auto make_tagged_tuple(Args&&... args)
    -> tagged_tuple<TagList, internal::special_decay_t<Args>...>
{
  return tagged_tuple<TagList, internal::special_decay_t<Args>...>{
      std::forward<Args>(args)...};
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

CAMP_SUPPRESS_HD_WARN
template <typename Fn, camp::idx_t... Sequence, typename TupleLike>
CAMP_HOST_DEVICE constexpr auto invoke_with_order(TupleLike&& t,
                                                  Fn&& f,
                                                  camp::idx_seq<Sequence...>)
    -> decltype(f(get<Sequence>(t)...))
{
  return f(get<Sequence>(t)...);
}

CAMP_SUPPRESS_HD_WARN
template <typename Fn, typename TupleLike>
CAMP_HOST_DEVICE constexpr auto invoke(TupleLike&& t, Fn&& f) -> decltype(
    invoke_with_order(forward<TupleLike>(t),
                      forward<Fn>(f),
                      camp::make_idx_seq_t<tuple_size<TupleLike>::value>{}))
{
  return invoke_with_order(
      forward<TupleLike>(t),
      forward<Fn>(f),
      camp::make_idx_seq_t<tuple_size<TupleLike>::value>{});
}
}  // namespace camp

namespace internal
{
template <class Tuple, camp::idx_t... Idxs>
void print_tuple(std::ostream& os, Tuple const& t, camp::idx_seq<Idxs...>)
{
  camp::sink((void*)&(os << (Idxs == 0 ? "" : ", ") << camp::get<Idxs>(t))...);
}
}  // namespace internal

template <class... Args>
auto operator<<(std::ostream& os, camp::tuple<Args...> const& t)
    -> std::ostream&
{
  os << "(";
  internal::print_tuple(os, t, camp::make_idx_seq_t<sizeof...(Args)>{});
  return os << ")";
}


#endif /* camp_tuple_HPP__ */
