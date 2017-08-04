#ifndef __CAMP_HPP
#define __CAMP_HPP

#include <cstddef>
#include <type_traits>
#include <array>

namespace camp
{
// Types
using idx_t = std::ptrdiff_t;

// Fwd
template <typename... Ts>
struct list;
template <typename T>
struct as_list;
template<typename T>
struct as_array;
template<typename T>
struct size;

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
  static constexpr std::array<T,sizeof...(vs)> array() noexcept {
      return std::array<T,sizeof...(vs)>{vs...};
  }
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

template <class...Ts>
using idx_seq_for_t = typename make_idx_seq<sizeof...(Ts)>::type;

template <typename T>
struct idx_seq_from;

template <template <typename...> class T, typename... Args>
struct idx_seq_from<T<Args...>> : make_idx_seq<sizeof...(Args)>{
};

template <typename T, T... Args>
struct idx_seq_from<int_seq<T, Args...>> : make_idx_seq<sizeof...(Args)>{
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

template <template <typename...> class T, typename... Args>
struct as_list<T<Args...>> {
    using type = list<Args...>;
};

template <typename T, T... Args>
struct as_list<int_seq<T, Args...>> {
    using type = list<integral<T, Args>...>;
};

//// array: TODO, using std::array for now
template<typename T, T...Vals>
struct as_array<int_seq<T, Vals...>>{
    constexpr static std::array<T, sizeof...(Vals)>value{Vals...};
};

//// size
template<typename ...Args>
struct size<list<Args...>>{
    constexpr static idx_t value{sizeof...(Args)};
};

template<typename T, T...Args>
struct size<int_seq<T, Args...>>{
    constexpr static idx_t value{sizeof...(Args)};
};

}  // end namespace camp

#endif /* __CAMP_HPP */
