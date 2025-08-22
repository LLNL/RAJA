/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for RAJA Printing definitions.
 *
 *          Definitions in this file will propagate to all RAJA header files.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_Printing_HPP
#define RAJA_Printing_HPP

#include "RAJA/config.hpp"

#include <ostream>
#include <type_traits>

namespace RAJA
{

namespace detail
{

//! Printing helper class to add printability to types defined outside of RAJA.
//
//  Specialize to customize printing or add printing for a type. This avoids
//  conflicts if an operator<< is later added for the type in question.
//  Write specializations for non-const reference and const-reference.
template < typename T >
struct StreamInsertHelper
{
  static_assert(std::is_lvalue_reference_v<T>);

  T m_val;

  std::ostream& operator()(std::ostream& str) const
  {
    return str << m_val;
  }
};

// deduction guide for StreamInsertHelper
template<typename T>
StreamInsertHelper(T& val) -> StreamInsertHelper<T&>;
template<typename T>
StreamInsertHelper(T const& val) -> StreamInsertHelper<T const&>;
template<typename T>
StreamInsertHelper(T&& val) -> StreamInsertHelper<T const&>;
template<typename T>
StreamInsertHelper(T const&& val) -> StreamInsertHelper<T const&>;

// Allow printing of StreamInsertHelper using its call operator
template < typename T >
inline std::ostream& operator<<(std::ostream& str, StreamInsertHelper<T> const& si)
{
  return si(str);
}

}  // closing brace for detail namespace

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
