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
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
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
 * An expression that returns the value of the specified RAJA::kernel
 * parameter.
 *
 * This allows run-time values to affect the control logic within
 * RAJA::kernel execution policies.
 */
template <camp::idx_t ParamId>
struct Param : public internal::ParamBase {

  constexpr static camp::idx_t param_idx = ParamId;

  template <typename Data>
  RAJA_HOST_DEVICE RAJA_INLINE static auto eval(Data const &data)
      -> decltype(camp::get<ParamId>(data.param_tuple))
  {
    return camp::get<ParamId>(data.param_tuple);
  }
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


template <typename Condition, typename... EnclosedStmts>
struct StatementExecutor<statement::If<Condition, EnclosedStmts...>> {


  template <typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {

    if (Condition::eval(data)) {
      execute_statement_list<camp::list<EnclosedStmts...>>(
          std::forward<Data>(data));
    }
  }
};


}  // namespace internal
}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_HPP */
