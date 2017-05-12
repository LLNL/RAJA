include(CheckCXXSourceCompiles)

set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
if (NOT MSVC)
  set(CMAKE_REQUIRED_FLAGS "-std=c++11")
endif()

CHECK_CXX_SOURCE_COMPILES(
"#include <type_traits>
#include <limits>

template <typename T>
struct signed_limits {
  static constexpr T min()
  {
    return static_cast<T>(1llu << ((8llu * sizeof(T)) - 1llu));
  }
  static constexpr T max()
  {
    return static_cast<T>(~(1llu << ((8llu * sizeof(T)) - 1llu)));
  }
};

template <typename T>
struct unsigned_limits {
  static constexpr T min()
  {
    return static_cast<T>(0);
  }
  static constexpr T max()
  {
    return static_cast<T>(0xFFFFFFFFFFFFFFFF);
  }
};

template <typename T>
struct limits : public std::conditional<
  std::is_signed<T>::value,
  signed_limits<T>,
  unsigned_limits<T>>::type {
};

template <typename T>
void check() {
  static_assert(limits<T>::min() == std::numeric_limits<T>::min(), \"min failed\");
  static_assert(limits<T>::max() == std::numeric_limits<T>::max(), \"max failed\");
}

int main() {
  check<char>();
  check<unsigned char>();
  check<short>();
  check<unsigned short>();
  check<int>();
  check<unsigned int>();
  check<long>();
  check<unsigned long>();
  check<long int>();
  check<unsigned long int>();
  check<long long>();
  check<unsigned long long>();
}" check_power_of_two_integral_types)

set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})

if(NOT check_power_of_two_integral_types)
  message(FATAL_ERROR "RAJA fast limits are unsupported for your compiler/architecture")
endif()
