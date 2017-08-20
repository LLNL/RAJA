#ifndef CAMP_NUMBER_HPP
#define CAMP_NUMBER_HPP

#include <array>
#include <type_traits>

#include "camp/defines.hpp"
namespace camp
{

// TODO: document
template <typename T, T v>
struct integral : std::integral_constant<T, v> {
};

// TODO: document
template <idx_t N>
struct num : integral<idx_t, N> {
};

// TODO: document
struct false_t : num<false> {
};
// TODO: document
struct true_t : num<true> {
};

// TODO: document
template <typename T, T... vs>
struct int_seq {
  // TODO: factor this out, do not like
  static constexpr std::array<T, sizeof...(vs)> array() noexcept
  {
    return std::array<T, sizeof...(vs)>{vs...};
  }
  static constexpr idx_t size() noexcept { return sizeof...(vs); }
  using value_type = T;
  using type = int_seq;
};

// TODO: document
template <idx_t... vs>
using idx_seq = int_seq<idx_t, vs...>;

namespace detail
{
  template <typename T, typename N>
  struct gen_seq;
#if defined(CAMP_USE_MAKE_INTEGER_SEQ) && !__NVCC__
  template <typename T, T N>
  struct gen_seq<T, integral<T, N>> {
    using type = __make_integer_seq<int_seq, T, N>;
  };
#else
  template <typename T, typename S1, typename S2>
  struct concat;

  template <typename T, T... I1, T... I2>
  struct concat<T, int_seq<T, I1...>, int_seq<T, I2...>> {
    using type = typename int_seq<T, I1..., (sizeof...(I1) + I2)...>::type;
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

// TODO: document
template <idx_t Upper>
struct make_idx_seq {
  using type = typename detail::gen_seq<idx_t, integral<idx_t, Upper>>::type;
};


// TODO: document
template <idx_t Upper>
using make_idx_seq_t = typename make_idx_seq<Upper>::type;

#if defined(CAMP_TEST)
namespace test
{
  CHECK_TSAME((make_idx_seq_t<3>), (idx_seq<0, 1, 2>));
  CHECK_TSAME((make_idx_seq_t<2>), (idx_seq<0, 1>));
  CHECK_TSAME((make_idx_seq_t<1>), (idx_seq<0>));
  CHECK_TSAME((make_idx_seq_t<0>), (idx_seq<>));
}
#endif


// TODO: document
template <class... Ts>
using idx_seq_for_t = typename make_idx_seq<sizeof...(Ts)>::type;

// TODO: document
template <typename T>
struct idx_seq_from;

// TODO: document
template <template <typename...> class T, typename... Args>
struct idx_seq_from<T<Args...>> : make_idx_seq<sizeof...(Args)> {
};

// TODO: document
template <typename T, T... Args>
struct idx_seq_from<int_seq<T, Args...>> : make_idx_seq<sizeof...(Args)>::type {
};

// TODO: document
template <typename T>
using idx_seq_from_t = typename idx_seq_from<T>::type;

// TODO: document
template <typename T, T Upper>
struct make_int_seq : detail::gen_seq<T, integral<T, Upper>>::type {
};

// TODO: document
template <typename T, idx_t Upper>
using make_int_seq_t = typename make_int_seq<T, Upper>::type;

// TODO: document
template <typename Cond, typename Then, typename Else>
struct if_ {
  static_assert(Cond::value, "got true with a false val somehow");
  using type = Then;
};
template <typename Then, typename Else>
struct if_<std::false_type, Then, Else> {
  using type = Else;
};
template <typename Then, typename Else>
struct if_<num<0>, Then, Else> {
  using type = Else;
};
template <typename IT, typename Then, typename Else>
struct if_<integral<IT, 0>, Then, Else> {
  using type = Else;
};
template <typename IT, typename Then, typename Else>
struct if_<std::integral_constant<IT, 0>, Then, Else> {
  using type = Else;
};

template <idx_t Cond, typename Then, typename Else>
struct if_v {
  using type = Then;
};
template <typename Then, typename Else>
struct if_v<0, Then, Else> {
  using type = Else;
};

// TODO: document
template <typename T>
struct not_ {
  using type = typename if_<T, false_t, true_t>::type;
};

}  // end namespace camp

#endif /* CAMP_NUMBER_HPP */
