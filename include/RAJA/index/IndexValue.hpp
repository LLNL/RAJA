/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file for strongly-typed integer class.
 *
 ******************************************************************************
 */

#ifndef RAJA_INDEXVALUE_HPP
#define RAJA_INDEXVALUE_HPP

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

#include "RAJA/config.hpp"

#include "RAJA/util/defines.hpp"
#include "RAJA/util/types.hpp"

#include <string>

namespace RAJA
{

/*!
 * \brief Strongly typed "integer" class.
 *
 * Allows integers to be associated with a type, and disallows automatic
 * conversion.
 *
 * Useful for maintaining correctness in multidimensional loops and arrays.
 *
 * Use the RAJA_INDEX_VALUE(NAME) macro to define new indices.
 *
 * Yes, this uses the curiously-recurring template pattern.
 */
template <typename TYPE>
class IndexValue
{
public:
  /*!
   * \brief Default constructor initializes value to 0.
   */
  RAJA_HOST_DEVICE
  RAJA_INLINE
  constexpr IndexValue() : value(0) {}

  /*!
   * \brief Explicit constructor.
   * \param v   Initial value
   */
  RAJA_HOST_DEVICE
  RAJA_INLINE
  constexpr explicit IndexValue(Index_type v) : value(v) {}

  /*!
   * \brief Dereference provides cast-to-integer.
   */
  RAJA_HOST_DEVICE
  RAJA_INLINE
  Index_type &operator*(void) { return value; }

  /*!
   * \brief Dereference provides cast-to-integer.
   */
  RAJA_HOST_DEVICE
  RAJA_INLINE
  const Index_type &operator*(void)const { return value; }

  RAJA_HOST_DEVICE
  RAJA_INLINE
  TYPE &operator++(int)
  {
    value++;
    return *static_cast<TYPE *>(this);
  }

  RAJA_HOST_DEVICE
  RAJA_INLINE
  TYPE &operator++()
  {
    value++;
    return *static_cast<TYPE *>(this);
  }

  RAJA_HOST_DEVICE
  RAJA_INLINE
  TYPE &operator--(int)
  {
    value--;
    return *static_cast<TYPE *>(this);
  }

  RAJA_HOST_DEVICE
  RAJA_INLINE
  TYPE &operator--()
  {
    value--;
    return *static_cast<TYPE *>(this);
  }

  RAJA_HOST_DEVICE
  RAJA_INLINE
  TYPE operator+(Index_type a) const { return TYPE(value + a); }

  RAJA_HOST_DEVICE
  RAJA_INLINE
  TYPE operator+(TYPE a) const { return TYPE(value + a.value); }

  RAJA_HOST_DEVICE
  RAJA_INLINE
  TYPE operator-(Index_type a) const { return TYPE(value - a); }

  RAJA_HOST_DEVICE
  RAJA_INLINE
  TYPE operator-(TYPE a) const { return TYPE(value - a.value); }

  RAJA_HOST_DEVICE
  RAJA_INLINE
  TYPE operator*(Index_type a) const { return TYPE(value * a); }

  RAJA_HOST_DEVICE
  RAJA_INLINE
  TYPE operator*(TYPE a) const { return TYPE(value * a.value); }

  RAJA_HOST_DEVICE
  RAJA_INLINE
  TYPE operator/(Index_type a) const { return TYPE(value / a); }

  RAJA_HOST_DEVICE
  RAJA_INLINE
  TYPE operator/(TYPE a) const { return TYPE(value / a.value); }

  RAJA_HOST_DEVICE
  RAJA_INLINE TYPE &operator+=(Index_type x)
  {
    value += x;
    return *static_cast<TYPE *>(this);
  }

  RAJA_HOST_DEVICE
  RAJA_INLINE TYPE &operator+=(TYPE x)
  {
    value += x.value;
    return *static_cast<TYPE *>(this);
  }

  RAJA_HOST_DEVICE
  RAJA_INLINE TYPE &operator-=(Index_type x)
  {
    value -= x;
    return *static_cast<TYPE *>(this);
  }

  RAJA_HOST_DEVICE
  RAJA_INLINE TYPE &operator-=(TYPE x)
  {
    value -= x.value;
    return *static_cast<TYPE *>(this);
  }

  RAJA_HOST_DEVICE
  RAJA_INLINE TYPE &operator*=(Index_type x)
  {
    value *= x;
    return *static_cast<TYPE *>(this);
  }

  RAJA_HOST_DEVICE
  RAJA_INLINE TYPE &operator*=(TYPE x)
  {
    value *= x.value;
    return *static_cast<TYPE *>(this);
  }

  RAJA_HOST_DEVICE
  RAJA_INLINE TYPE &operator/=(Index_type x)
  {
    value /= x;
    return *static_cast<TYPE *>(this);
  }

  RAJA_HOST_DEVICE
  RAJA_INLINE TYPE &operator/=(TYPE x)
  {
    value /= x.value;
    return *static_cast<TYPE *>(this);
  }

  RAJA_HOST_DEVICE
  RAJA_INLINE bool operator<(Index_type x) const { return (value < x); }

  RAJA_HOST_DEVICE
  RAJA_INLINE bool operator<(TYPE x) const { return (value < x.value); }

  RAJA_HOST_DEVICE
  RAJA_INLINE bool operator<=(Index_type x) const { return (value <= x); }

  RAJA_HOST_DEVICE
  RAJA_INLINE bool operator<=(TYPE x) const { return (value <= x.value); }

  RAJA_HOST_DEVICE
  RAJA_INLINE bool operator>(Index_type x) const { return (value > x); }

  RAJA_HOST_DEVICE
  RAJA_INLINE bool operator>(TYPE x) const { return (value > x.value); }

  RAJA_HOST_DEVICE
  RAJA_INLINE bool operator>=(Index_type x) const { return (value >= x); }

  RAJA_HOST_DEVICE
  RAJA_INLINE bool operator>=(TYPE x) const { return (value >= x.value); }

  RAJA_HOST_DEVICE
  RAJA_INLINE bool operator==(Index_type x) const { return (value == x); }

  RAJA_HOST_DEVICE
  RAJA_INLINE bool operator==(TYPE x) const { return (value == x.value); }

  RAJA_HOST_DEVICE
  RAJA_INLINE bool operator!=(Index_type x) const { return (value != x); }

  RAJA_HOST_DEVICE
  RAJA_INLINE bool operator!=(TYPE x) const { return (value != x.value); }

  // This is not implemented... but should be by the derived type
  // this is done by the macro
  static std::string getName(void);

protected:
  Index_type value;
};


/*!
 * \brief Function provides a way to take either an int or any Index<> type, and
 * convert it to another type, possibly another Index or an int.
 *
 * This has been refactored to use SFINAE to differentiate between IndexValue
 * derived types an non-IndexValue types, and return the correct type
 * automatically
 */

template <typename TO, typename FROM>
RAJA_HOST_DEVICE RAJA_INLINE TO convertIndex_helper(FROM val)
{
  return TO{val};
}
template <typename TO, typename FROM>
RAJA_HOST_DEVICE RAJA_INLINE TO
convertIndex_helper(typename FROM::IndexValueType val)
{
  return static_cast<TO>(*val);
}

template <typename TO, typename FROM>
RAJA_HOST_DEVICE RAJA_INLINE TO convertIndex(FROM val)
{
  return convertIndex_helper<TO, FROM>(val);
}


}  // namespace RAJA

/*!
 * \brief Helper Macro to create new Index types.
 */
#define RAJA_INDEX_VALUE(TYPE, NAME)                                           \
  class TYPE : public RAJA::IndexValue<TYPE>                                   \
  {                                                                            \
  public:                                                                      \
    typedef TYPE IndexValueType;                                               \
    RAJA_HOST_DEVICE RAJA_INLINE TYPE() : RAJA::IndexValue<TYPE>::IndexValue() \
    {                                                                          \
    }                                                                          \
    RAJA_HOST_DEVICE RAJA_INLINE explicit TYPE(RAJA::Index_type v)             \
        : RAJA::IndexValue<TYPE>::IndexValue(v)                                \
    {                                                                          \
    }                                                                          \
    static inline std::string getName(void) { return NAME; }                   \
  };

#endif
