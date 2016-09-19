/*!
0;95;0c
******************************************************************************
 *
 * \file
 *
 * \brief   Header file for RAJA operator definitions
 *
 *          Definitions in this file will propagate to all RAJA header files.
 *
 ******************************************************************************
 */

#ifndef RAJA_operators_HXX
#define RAJA_operators_HXX

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
// For additional details, please also read raja/README-license.txt.
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

#include "RAJA/config.hxx"
#include "RAJA/internal/defines.hxx"

#include <cfloat>
#include <cstdint>

namespace RAJA
{

namespace operators
{

namespace detail
{

struct associative_tag {
};

template <typename Arg1, typename Arg2, typename Result>
struct binary_function {
  using first_argument_type = Arg1;
  using second_argument_type = Arg2;
  using result_type = Result;
};

template <typename Argument, typename Result>
struct unary_function {
  using argument_type = Argument;
  using result_type = Result;
};

template <typename T>
RAJA_HOST_DEVICE constexpr T min()
{
  return 0;
};
template <>
RAJA_HOST_DEVICE constexpr int16_t min<int16_t>()
{
  return 0xFFFF;
}
template <>
RAJA_HOST_DEVICE constexpr uint16_t min<uint16_t>()
{
  return 0x0000;
}
template <>
RAJA_HOST_DEVICE constexpr int32_t min<int32_t>()
{
  return 0xFFFFFFFF;
}
template <>
RAJA_HOST_DEVICE constexpr uint32_t min<uint32_t>()
{
  return 0x00000000;
}
template <>
RAJA_HOST_DEVICE constexpr int64_t min<int64_t>()
{
  return 0xFFFFFFFFFFFFFFFF;
}
template <>
RAJA_HOST_DEVICE constexpr uint64_t min<uint64_t>()
{
  return 0x0000000000000000;
}
template <>
RAJA_HOST_DEVICE constexpr float min<float>()
{
  return -FLT_MAX;
}
template <>
RAJA_HOST_DEVICE constexpr double min<double>()
{
  return -DBL_MAX;
}
template <>
RAJA_HOST_DEVICE constexpr long double min<long double>()
{
  return -LDBL_MAX;
}

template <typename T>
RAJA_HOST_DEVICE constexpr T max()
{
  return 0;
};
template <>
RAJA_HOST_DEVICE constexpr int16_t max<int16_t>()
{
  return 0x7FFF;
}
template <>
RAJA_HOST_DEVICE constexpr uint16_t max<uint16_t>()
{
  return 0xFFFF;
}
template <>
RAJA_HOST_DEVICE constexpr int32_t max<int32_t>()
{
  return 0x7FFFFFFF;
}
template <>
RAJA_HOST_DEVICE constexpr uint32_t max<uint32_t>()
{
  return 0xFFFFFFFF;
}
template <>
RAJA_HOST_DEVICE constexpr int64_t max<int64_t>()
{
  return 0x7FFFFFFFFFFFFFFF;
}
template <>
RAJA_HOST_DEVICE constexpr uint64_t max<uint64_t>()
{
  return 0xFFFFFFFFFFFFFFFF;
}
template <>
RAJA_HOST_DEVICE constexpr float max<float>()
{
  return FLT_MAX;
}
template <>
RAJA_HOST_DEVICE constexpr double max<double>()
{
  return DBL_MAX;
}
template <>
RAJA_HOST_DEVICE constexpr long double max<long double>()
{
  return LDBL_MAX;
}
}

// Arithmetic

template <typename T>
struct plus : public detail::binary_function<T, T, T>, detail::associative_tag {
  RAJA_HOST_DEVICE T operator()(const T& lhs, const T& rhs)
  {
    return lhs + rhs;
  }
  static constexpr const T identity = T{0};
};

template <typename T>
struct minus : public detail::binary_function<T, T, T> {
  RAJA_HOST_DEVICE T operator()(const T& lhs, const T& rhs)
  {
    return lhs - rhs;
  }
};

template <typename T>
struct multiplies : public detail::binary_function<T, T, T>,
                    detail::associative_tag {
  RAJA_HOST_DEVICE T operator()(const T& lhs, const T& rhs)
  {
    return lhs * rhs;
  }
  static constexpr const T identity = T{1};
};

template <typename T>
struct divides : public detail::binary_function<T, T, T> {
  RAJA_HOST_DEVICE T operator()(const T& lhs, const T& rhs)
  {
    return lhs / rhs;
  }
};

template <typename T>
struct modulus : public detail::binary_function<T, T, T> {
  RAJA_HOST_DEVICE T operator()(const T& lhs, const T& rhs)
  {
    return lhs % rhs;
  }
};

// conditions

template <typename T>
struct logical_and : public detail::binary_function<T, T, bool>,
                     detail::associative_tag {
  RAJA_HOST_DEVICE bool operator()(const T& lhs, const T& rhs)
  {
    return lhs && rhs;
  }
  static constexpr const T identity = T{true};
};

template <typename T>
struct logical_or : public detail::binary_function<T, T, bool>,
                    detail::associative_tag {
  RAJA_HOST_DEVICE bool operator()(const T& lhs, const T& rhs)
  {
    return lhs || rhs;
  }
  static constexpr const T identity = T{false};
};

template <typename T>
struct logical_not : public detail::unary_function<T, bool> {
  RAJA_HOST_DEVICE bool operator()(const T& lhs) { return !lhs; }
};

// bitwise

template <typename T>
struct bit_or : public detail::binary_function<T, T, T> {
  RAJA_HOST_DEVICE bool operator()(const T& lhs, const T& rhs)
  {
    return lhs | rhs;
  }
};

template <typename T>
struct bit_and : public detail::binary_function<T, T, T> {
  RAJA_HOST_DEVICE bool operator()(const T& lhs, const T& rhs)
  {
    return lhs & rhs;
  }
};

template <typename T>
struct bit_xor : public detail::binary_function<T, T, T> {
  RAJA_HOST_DEVICE T operator()(const T& lhs, const T& rhs)
  {
    return lhs ^ rhs;
  }
};

// comparison

template <typename T>
struct minimum : public detail::binary_function<T, T, T>,
                 detail::associative_tag {
  RAJA_HOST_DEVICE T operator()(const T& lhs, const T& rhs)
  {
    return (lhs < rhs) ? lhs : rhs;
  }
  static constexpr const T identity = detail::max<T>();
};

template <typename T>
struct maximum : public detail::binary_function<T, T, T>,
                 detail::associative_tag {
  RAJA_HOST_DEVICE T operator()(const T& lhs, const T& rhs)
  {
    return (lhs < rhs) ? rhs : lhs;
  }
  static constexpr const T identity = detail::min<T>();
};

// logical comparison

template <typename T>
struct equal_to : public detail::binary_function<T, T, bool> {
  RAJA_HOST_DEVICE bool operator()(const T& lhs, const T& rhs)
  {
    return lhs == rhs;
  }
};

template <typename T>
struct not_equal_to : public detail::binary_function<T, T, bool> {
  RAJA_HOST_DEVICE bool operator()(const T& lhs, const T& rhs)
  {
    return lhs != rhs;
  }
};

template <typename T>
struct greater : public detail::binary_function<T, T, bool> {
  RAJA_HOST_DEVICE bool operator()(const T& lhs, const T& rhs)
  {
    return lhs > rhs;
  }
};

template <typename T>
struct less : public detail::binary_function<T, T, bool> {
  RAJA_HOST_DEVICE bool operator()(const T& lhs, const T& rhs)
  {
    return lhs < rhs;
  }
};

template <typename T>
struct greater_equal : public detail::binary_function<T, T, bool> {
  RAJA_HOST_DEVICE bool operator()(const T& lhs, const T& rhs)
  {
    return lhs > rhs;
  }
};

template <typename T>
struct less_equal : public detail::binary_function<T, T, bool> {
  RAJA_HOST_DEVICE bool operator()(const T& lhs, const T& rhs)
  {
    return lhs < rhs;
  }
};

// filters

template <typename T>
struct identity : public detail::unary_function<T, T> {
  RAJA_HOST_DEVICE T operator()(const T& lhs) { return lhs; }
};

template <typename T, typename U>
struct project1st : public detail::binary_function<T, U, T> {
  RAJA_HOST_DEVICE T operator()(const T& lhs, const U& rhs) { return lhs; }
};

template <typename T, typename U = T>
struct project2nd : public detail::binary_function<T, U, U> {
  RAJA_HOST_DEVICE U operator()(const T& lhs, const U& rhs) { return rhs; }
};

}  // closing brace for operators namespace

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
