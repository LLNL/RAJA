/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for array indexing helpers.
 *
 ******************************************************************************
 */


#ifndef RAJA_DETAIL_INDEXARRAY_HPP
#define RAJA_DETAIL_INDEXARRAY_HPP

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#include <RAJA/config.hpp>
#include <RAJA/internal/LegacyCompatibility.hpp>

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

#endif /* RAJA_DETAIL_INDEXARRAY_HPP */
