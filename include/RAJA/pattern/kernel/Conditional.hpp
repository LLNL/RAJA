/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for kernel conditional templates
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_kernel_Conditional_HPP
#define RAJA_pattern_kernel_Conditional_HPP


#include "RAJA/config.hpp"

#include "RAJA/pattern/kernel/internal.hpp"

#include <iostream>
#include <type_traits>

namespace RAJA
{
namespace statement
{


/*!
 * A RAJA::kernel statement that implements conditional control logic
 *
 */
template <typename Condition, typename... EnclosedStmts>
struct If : public internal::Statement<camp::nil, EnclosedStmts...> {
};


/*!
 * An expression that returns a compile time literal value.
 *
 */
template <long value>
struct Value {

  template <typename Data>
  RAJA_HOST_DEVICE RAJA_INLINE static long eval(Data const &)
  {
    return value;
  }
};

/*!
 * An equality expression
 *
 */
template <typename L, typename R>
struct Equals {

  template <typename Data>
  RAJA_HOST_DEVICE RAJA_INLINE static bool eval(Data const &data)
  {
    return L::eval(data) == R::eval(data);
  }
};

/*!
 * A negated equality expression
 *
 */
template <typename L, typename R>
struct NotEquals {

  template <typename Data>
  RAJA_HOST_DEVICE RAJA_INLINE static bool eval(Data const &data)
  {
    return L::eval(data) != R::eval(data);
  }
};


/*!
 * A logical OR expression
 *
 */
template <typename L, typename R>
struct Or {

  template <typename Data>
  RAJA_HOST_DEVICE RAJA_INLINE static bool eval(Data const &data)
  {
    return L::eval(data) || R::eval(data);
  }
};


/*!
 * A logical AND expression
 *
 */
template <typename L, typename R>
struct And {

  template <typename Data>
  RAJA_HOST_DEVICE RAJA_INLINE static bool eval(Data const &data)
  {
    return L::eval(data) && R::eval(data);
  }
};


/*!
 * A less than expression
 *
 */
template <typename L, typename R>
struct LessThan {

  template <typename Data>
  RAJA_HOST_DEVICE RAJA_INLINE static bool eval(Data const &data)
  {
    return L::eval(data) < R::eval(data);
  }
};


/*!
 * A less or equals than expression
 *
 */
template <typename L, typename R>
struct LessThanEq {

  template <typename Data>
  RAJA_HOST_DEVICE RAJA_INLINE static bool eval(Data const &data)
  {
    return L::eval(data) <= R::eval(data);
  }
};


/*!
 * A greater than expression
 *
 */
template <typename L, typename R>
struct GreaterThan {

  template <typename Data>
  RAJA_HOST_DEVICE RAJA_INLINE static bool eval(Data const &data)
  {
    return L::eval(data) > R::eval(data);
  }
};


/*!
 * A greater or equals than expression
 *
 */
template <typename L, typename R>
struct GreaterThanEq {

  template <typename Data>
  RAJA_HOST_DEVICE RAJA_INLINE static bool eval(Data const &data)
  {
    return L::eval(data) >= R::eval(data);
  }
};


/*!
 * A negation expression
 *
 */
template <typename L>
struct Not {

  template <typename Data>
  RAJA_HOST_DEVICE RAJA_INLINE static bool eval(Data const &data)
  {
    return !(L::eval(data));
  }
};


}  // end namespace statement

namespace internal
{


template <typename Condition, typename... EnclosedStmts, typename Types>
struct StatementExecutor<statement::If<Condition, EnclosedStmts...>, Types> {


  template <typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {

    if (Condition::eval(data)) {
      execute_statement_list<camp::list<EnclosedStmts...>, Types>(
          std::forward<Data>(data));
    }
  }
};


}  // namespace internal
}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_HPP */
