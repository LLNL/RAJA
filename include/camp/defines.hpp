#ifndef CAMP_DEFINES_HPP
#define CAMP_DEFINES_HPP

#include <cstddef>
#include <cstdint>

namespace camp
{

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

#if defined(__has_builtin)
#if __has_builtin(__make_integer_seq)
#define CAMP_USE_MAKE_INTEGER_SEQ 1
#endif
#endif

// Types
using idx_t = std::ptrdiff_t;

// Helper macros
#define CAMP_MAKE_L(X)                                             \
  template <typename Lambda, typename... Rest>                     \
  struct X##_l {                                                   \
    using type = typename X<Lambda::template expr, Rest...>::type; \
  };


#if defined(CAMP_TEST)
template <typename T1, typename T2>
struct AssertSame {
  static_assert(std::is_same<T1, T2>::value,
                "is_same assertion failed <see below for more information>");
  static bool constexpr value = std::is_same<T1, T2>::value;
};
#define UNQUOTE(...) __VA_ARGS__
#define CHECK_SAME(X, Y) \
  static_assert(AssertSame<UNQUOTE X, UNQUOTE Y>::value, #X " same as " #Y)
#define CHECK_TSAME(X, Y)                                               \
  static_assert(AssertSame<typename UNQUOTE X::type, UNQUOTE Y>::value, \
                #X " same as " #Y)
template <typename Assertion, idx_t i>
struct AssertValue {
  static_assert(Assertion::value == i,
                "value assertion failed <see below for more information>");
  static bool const value = Assertion::value == i;
};
#define CHECK_IEQ(X, Y) \
  static_assert(AssertValue<UNQUOTE X, UNQUOTE Y>::value, #X "::value == " #Y)
#endif

}

#endif /*  */
