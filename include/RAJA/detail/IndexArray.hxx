#ifndef RAJA_DETAIL_INDEXARRAY_HXX
#define RAJA_DETAIL_INDEXARRAY_HXX

#include <RAJA/LegacyCompatibility.hxx>

namespace RAJA
{
namespace detail
{
template <size_t Offset, typename Type>
struct index_storage {
  RAJA_HOST_DEVICE
  RAJA_INLINE
  Type& get() { return data; }
  RAJA_HOST_DEVICE
  RAJA_INLINE
  constexpr const Type& get() const { return data; }
  Type data;
};

template <typename StorageType>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto get_data(StorageType& s)
    -> decltype(s.get())
{
  return s.get();
}
template <typename StorageType>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto get_data(const StorageType& s)
    -> decltype(s.get())
{
  return s.get();
}

template <size_t I, typename AType_in>
struct select_element {
  using AType = typename std::remove_reference<AType_in>::type;
  using return_type = typename AType::type&;
  using const_return_type = const typename AType::type&;
  using value_type = typename AType::type;

  RAJA_HOST_DEVICE
  RAJA_INLINE
  static constexpr return_type get(AType_in& a, size_t offset)
  {
    return (offset == I) ? get_data<index_storage<I, value_type>>(a)
                         : select_element<I - 1, AType_in>::get(a, offset);
  }
  RAJA_HOST_DEVICE
  RAJA_INLINE
  static constexpr const_return_type get(const AType_in& a, size_t offset)
  {
    return (offset == I) ? get_data<const index_storage<I, value_type>>(a)
                         : select_element<I - 1, AType_in>::get(a, offset);
  }
};

template <typename AType_in>
struct select_element<0, AType_in> {
  using AType = typename std::remove_reference<AType_in>::type;
  using return_type = typename AType::type&;
  using const_return_type = const typename AType::type&;
  using value_type = typename AType::type;

  RAJA_HOST_DEVICE
  RAJA_INLINE
  static constexpr return_type get(AType_in& a, size_t offset)
  {
    return get_data<index_storage<0, value_type>>(a);
  }

  RAJA_HOST_DEVICE
  RAJA_INLINE
  static constexpr const_return_type get(const AType_in& a, size_t offset)
  {
    return get_data<const index_storage<0, value_type>>(a);
  }
};

template <typename... Types>
struct index_array_helper;

template <typename Type, size_t... orest>
struct index_array_helper<Type, VarOps::index_sequence<orest...>>
    : index_storage<orest, Type>... {
  using type = Type;
  using my_type = index_array_helper<Type, VarOps::index_sequence<orest...>>;
  static constexpr size_t size = sizeof...(orest);

  RAJA_HOST_DEVICE
  RAJA_INLINE
  // constexpr : c++14 only
  Type& operator[](size_t offset)
  {
    return select_element<size - 1, my_type>::get(*this, offset);
  }

  RAJA_HOST_DEVICE
  RAJA_INLINE
  constexpr const Type& operator[](size_t offset) const
  {
    return select_element<size - 1, my_type>::get(*this, offset);
  }
};

template <typename Type, size_t... orest>
constexpr size_t
    index_array_helper<Type, VarOps::index_sequence<orest...>>::size;
}

template <size_t Size, typename Type>
struct index_array
    : public detail::index_array_helper<Type,
                                        VarOps::make_index_sequence<Size>> {
  static_assert(Size > 0, "index_arrays must have at least one element");
  using base =
      detail::index_array_helper<Type, VarOps::make_index_sequence<Size>>;
  using base::index_array_helper;
  using base::operator[];
};

template <size_t Offset, typename Type>
RAJA_HOST_DEVICE RAJA_INLINE Type& get(detail::index_storage<Offset, Type>& s)
{
  return s.data;
}

template <size_t Offset, typename Type>
RAJA_HOST_DEVICE RAJA_INLINE const Type& get(
    const detail::index_storage<Offset, Type>& s)
{
  return s.data;
}

namespace detail
{
template <typename Type, size_t... Seq, typename... Args>
RAJA_HOST_DEVICE RAJA_INLINE auto make_index_array_helper(
    VarOps::index_sequence<Seq...>,
    Args... args) -> index_array<sizeof...(args), Type>
{
  index_array<sizeof...(args), Type> arr{};
  VarOps::ignore_args((get<Seq>(arr) = args)...);
  return arr;
};
}

template <typename Arg1, typename... Args>
RAJA_HOST_DEVICE RAJA_INLINE auto make_index_array(Arg1 arg1, Args... args)
    -> index_array<sizeof...(args) + 1, Arg1>
{
  return detail::make_index_array_helper<Arg1>(
      VarOps::make_index_sequence<sizeof...(args) + 1>(), arg1, args...);
};

template <size_t Size, typename Type>
std::ostream& operator<<(std::ostream& os, index_array<Size, Type> const& a)
{
  // const detail::index_array_helper<Type, VarOps::make_index_sequence<Size>> &
  // ah = a;
  // os << "array templated iteration: " << ah << std::endl;
  // os << "array runtime operator iteration: ";
  os << '[';
  for (size_t i = 0; i < Size - 1; ++i)
    os << a[i] << ", ";
  if (Size - 1 > 0) os << a[Size - 1];
  os << ']';
  return os;
}
}

#endif /* RAJA_DETAIL_INDEXARRAY_HXX */
