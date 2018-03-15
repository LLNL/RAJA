#ifndef RAJA_pattern_kernel_Conditional_HPP
#define RAJA_pattern_kernel_Conditional_HPP


#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

namespace RAJA
{
namespace statement
{


/*!
 * A kernel::forall statement that implements conditional control logic
 *
 */
template <typename Condition, typename... EnclosedStmts>
struct If : public internal::Statement<camp::nil, EnclosedStmts...> {
};


/*!
 * An expression that returns the value of the specified kernel::forall
 * parameter.
 *
 * This allows run-time values to affect the control logic within
 * kernel::forall policies.
 */
template <camp::idx_t ParamId>
struct Param {

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
 * An negation expression
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
