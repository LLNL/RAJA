#ifndef CAMP_HELPERS_HPP
#define CAMP_HELPERS_HPP

namespace camp
{

template <typename T>
T* declptr();

template <typename... Ts>
void sink(Ts...)
{
}

namespace ref
{
  template <class T>
  struct rem_s {
    using type = T;
  };
  template <class T>
  struct rem_s<T&> {
    using type = T;
  };
  template <class T>
  struct rem_s<T&&> {
    using type = T;
  };

  template <class T>
  using rem = typename rem_s<T>::type;
}  // end namespace ref

template <class T>
RAJA_HOST_DEVICE RAJA_INLINE constexpr T&& forward(ref::rem<T>& t) noexcept
{
  return static_cast<T&&>(t);
}
template <class T>
RAJA_HOST_DEVICE RAJA_INLINE constexpr T&& forward(ref::rem<T>&& t) noexcept
{
  return static_cast<T&&>(t);
}

}

#endif /* CAMP_HELPERS_HPP */
