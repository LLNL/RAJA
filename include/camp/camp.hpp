#ifndef __CAMP_HPP
#define __CAMP_HPP

#include <array>
#include <cstddef>
#include <type_traits>

namespace camp
{
// Defines
#if defined(__cpp_constexpr) && __cpp_constexpr >= 201304
#define CAMP_HAS_CONSTEXPR14
#define CAMP_CONSTEXPR14 constexpr
#else
#define CAMP_CONSTEXPR14
#endif

#if defined(__CUDACC__)
#define CAMP_DEVICE __device__
#define CAMP_HOST_DEVICE __host__ __device__
#else
#define CAMP_DEVICE
#define CAMP_HOST_DEVICE
#endif


// Types
using idx_t = std::ptrdiff_t;

// Fwd
template <typename... Ts>
struct list;
template <typename T>
struct as_list;
template <typename T>
struct as_array;
template <typename T>
struct size;

// helpers

template <typename T>
T* declptr();

template <typename... Ts>
void sink(Ts... args)
{
}

// Value

namespace detail
{
  struct nil;
}
template <typename val = detail::nil>
struct value {
  using type = val;
};

// Numbers

template <typename T, T v>
struct integral : std::integral_constant<T, v> {
};


template <idx_t N>
struct num : integral<idx_t, N> {
};

struct false_t : num<false> {
};
struct true_t : num<false> {
};

// Sequences
//// int_seq and idx_seq
template <typename T, T... vs>
struct int_seq {
  static constexpr std::array<T, sizeof...(vs)> array() noexcept
  {
    return std::array<T, sizeof...(vs)>{vs...};
  }
  static constexpr idx_t size() noexcept { return sizeof...(vs); }
  using value_type = T;
  using type = int_seq;
};

template <idx_t... vs>
using idx_seq = int_seq<idx_t, vs...>;

namespace integer_sequence_detail
{
  template <typename T, typename N>
  struct gen_seq;
#if defined(__has_builtin) && __has_builtin(__make_integer_seq)
  template <typename T, T N>
  struct gen_seq<T, integral<T, N>> : __make_integer_seq<int_seq, T, N>::type {
  };
#else
  template <typename T, typename S1, typename S2>
  struct concat;

  template <typename T, T... I1, T... I2>
  struct concat<T, list<integral<T, I1>...>, list<integral<T, I2>...>>
      : int_seq<T, I1..., (sizeof...(I1) + I2)...> {
  };

  template <typename T, typename N_t>
  struct gen_seq
      : concat<T,
               typename gen_seq<T, integral<T, N_t::value / 2>>::type,
               typename gen_seq<T, integral<T, N_t::value - N_t::value / 2>>::
                   type>::type {
  };

  template <typename T>
  struct gen_seq<T, integral<T, 0>> : int_seq<T> {
  };
  template <typename T>
  struct gen_seq<T, integral<T, 1>> : int_seq<T, 0> {
  };
#endif
}

template <idx_t Upper>
struct make_idx_seq
    : integer_sequence_detail::gen_seq<idx_t, integral<idx_t, Upper>>::type {
};

template <idx_t Upper>
using make_idx_seq_t = typename make_idx_seq<Upper>::type;

template <class... Ts>
using idx_seq_for_t = typename make_idx_seq<sizeof...(Ts)>::type;

template <typename T>
struct idx_seq_from;

template <template <typename...> class T, typename... Args>
struct idx_seq_from<T<Args...>> : make_idx_seq<sizeof...(Args)> {
};

template <typename T, T... Args>
struct idx_seq_from<int_seq<T, Args...>> : make_idx_seq<sizeof...(Args)> {
};

template <typename T>
using idx_seq_from_t = typename idx_seq_from<T>::type;

template <typename T, T Upper>
struct make_int_seq
    : integer_sequence_detail::gen_seq<T, integral<T, Upper>>::type {
};

template <typename T, idx_t Upper>
using make_int_seq_t = typename make_int_seq<T, Upper>::type;

//// list

template <typename... Ts>
struct list {
  using type = list;
};

namespace detail
{
  // Lookup from metal::at machinery
  template <idx_t, typename>
  struct entry {
  };

  template <typename, typename>
  struct entries;

  template <idx_t... keys, typename... vals>
  struct entries<idx_seq<keys...>, list<vals...>> : entry<keys, vals>... {
  };

  template <idx_t key, typename val>
  value<val> _lookup_impl(entry<key, val>*);

  template <typename>
  value<> _lookup_impl(...);

  template <typename vals, typename indices, idx_t Idx>
  struct _lookup
      : decltype(_lookup_impl<Idx>(declptr<entries<indices, vals>>())) {
  };

  template <typename T, idx_t Idx>
  struct get_at;
  template <typename T, idx_t Idx>
  struct get_at : _lookup<T, typename idx_seq_from<T>::type, Idx>::type {
  };
  template <template <class...> class T, typename X, typename... Rest>
  struct get_at<T<X, Rest...>, 0> {
    using type = X;
  };
  template <template <class...> class T,
            typename X,
            typename Y,
            typename... Rest>
  struct get_at<T<X, Y, Rest...>, 1> {
    using type = Y;
  };
}

template <typename T, typename U>
struct at;
template <typename T, idx_t Val>
struct at<T, num<Val>> : detail::get_at<T, Val> {
};

template <typename T, idx_t Idx>
using at_t = typename at<T, num<Idx>>::type;

template <template <typename...> class T, typename... Args>
struct as_list<T<Args...>> {
  using type = list<Args...>;
};

template <typename T, T... Args>
struct as_list<int_seq<T, Args...>> {
  using type = list<integral<T, Args>...>;
};

//// array: TODO, using std::array for now
template <typename T, T... Vals>
struct as_array<int_seq<T, Vals...>> {
  constexpr static std::array<T, sizeof...(Vals)> value{Vals...};
};

//// size
template <typename... Args>
struct size<list<Args...>> {
  constexpr static idx_t value{sizeof...(Args)};
};

template <typename T, T... Args>
struct size<int_seq<T, Args...>> {
  constexpr static idx_t value{sizeof...(Args)};
};

}  // end namespace camp

#endif /* __CAMP_HPP */
