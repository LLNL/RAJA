/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining atomic operations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
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

#ifndef RAJA_pattern_atomic_HPP
#define RAJA_pattern_atomic_HPP

#include "RAJA/config.hpp"
#include "RAJA/policy/atomic_auto.hpp"
#include "RAJA/policy/atomic_builtin.hpp"
#include "RAJA/util/defines.hpp"

namespace RAJA
{
namespace atomic
{


/*!
 * \file
 * Atomic operation functions in the namespace RAJA::atomic
 *
 * The dispatch of all of these is:
 *
 * T atomicAdd<Policy>(T *acc, T value)      -- User facing API
 *
 * calls
 *
 * T atomicAdd(Policy{}, T *acc, T value)    -- Policy specific implementation
 *
 *
 * With the exception of the auto_atomic policy which then calls the
 * "appropriate" policy implementation.
 *
 *
 * Current supported policies include:
 *
 *   auto_atomic       -- Attempts to do "the right thing"
 *
 *   cuda_atomic       -- Only atomic supported in CUDA device functions
 *
 *   omp_atomic        -- Available (and default) when OpenMP is enabled
 *                        these are safe inside and outside of OMP parallel
 *                        regions
 *
 *   builtin_atomic    -- Use the (nonstandard) __sync_fetch_and_XXX functions
 *
 *   seq_atomic        -- Non-atomic, does an unprotected (raw) operation
 *
 *
 * Current supported data types include:
 *
 *   32-bit and 64-bit integral types:
 *      -Native atomic support for "unsigned", "long", "unsigned long long" and
 *      "long long"
 *
 *      -General support, via CAS algorithm, for any 32-bit or 64-bit datatype
 *
 *   32-bit and 64-bit floating point types:  float and double
 *
 *
 * The implementation code lives in:
 * RAJA/policy/atomic_auto.hpp     -- for auto_atomic
 * RAJA/policy/atomic_builtin.hpp  -- for builtin_atomic
 * RAJA/policy/XXX/atomic.hpp      -- for omp_atomic, cuda_atomic, etc.
 *
 */


/*!
 * @brief Atomic add
 * @param acc Pointer to location of result value
 * @param value Value to add to *acc
 * @return Returns value at acc immediately before this operation completed
 */
RAJA_SUPPRESS_HD_WARN
template <typename Policy, typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicAdd(T volatile *acc, T value)
{
  return RAJA::atomic::atomicAdd(Policy{}, acc, value);
}


/*!
 * @brief Atomic subtract
 * @param acc Pointer to location of result value
 * @param value Value to subtract from *acc
 * @return Returns value at acc immediately before this operation completed
 */
RAJA_SUPPRESS_HD_WARN
template <typename Policy, typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicSub(T volatile *acc, T value)
{
  return RAJA::atomic::atomicSub(Policy{}, acc, value);
}


/*!
 * @brief Atomic minimum equivalent to (*acc) = std::min(*acc, value)
 * @param acc Pointer to location of result value
 * @param value Value to compare to *acc
 * @return Returns value at acc immediately before this operation completed
 */
RAJA_SUPPRESS_HD_WARN
template <typename Policy, typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicMin(T volatile *acc, T value)
{
  return RAJA::atomic::atomicMin(Policy{}, acc, value);
}


/*!
 * @brief Atomic maximum equivalent to (*acc) = std::max(*acc, value)
 * @param acc Pointer to location of result value
 * @param value Value to compare to *acc
 * @return Returns value at acc immediately before this operation completed
 */
RAJA_SUPPRESS_HD_WARN
template <typename Policy, typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicMax(T volatile *acc, T value)
{
  return RAJA::atomic::atomicMax(Policy{}, acc, value);
}


/*!
 * @brief Atomic increment
 * @param acc Pointer to location of value to increment
 * @return Returns value at acc immediately before this operation completed
 */
RAJA_SUPPRESS_HD_WARN
template <typename Policy, typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicInc(T volatile *acc)
{
  return RAJA::atomic::atomicInc(Policy{}, acc);
}


/*!
 * @brief Atomic increment with bound
 * Equivalent to *acc = ((*acc >= compare) ? 0 : ((*acc)+1))
 * This is for compatability with the CUDA atomicInc.
 * @param acc Pointer to location of value to increment
 * @param compare Bound value
 * @return Returns value at acc immediately before this operation completed
 */
RAJA_SUPPRESS_HD_WARN
template <typename Policy, typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicInc(T volatile *acc, T compare)
{
  return RAJA::atomic::atomicInc(Policy{}, acc, compare);
}


/*!
 * @brief Atomic decrement
 * @param acc Pointer to location of value to decrement
 * @return Returns value at acc immediately before this operation completed
 */
RAJA_SUPPRESS_HD_WARN
template <typename Policy, typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicDec(T volatile *acc)
{
  return RAJA::atomic::atomicDec(Policy{}, acc);
}


/*!
 * @brief Atomic decrement with bound
 * Equivalent to *acc = (((*acc==0)|(*acc>compare))?compare:((*acc)-1))
 * This is for compatability with the CUDA atomicDec.
 * @param acc Pointer to location of value to decrement
 * @param compare Bound value
 * @return Returns value at acc immediately before this operation completed
 */
RAJA_SUPPRESS_HD_WARN
template <typename Policy, typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicDec(T volatile *acc, T compare)
{
  return RAJA::atomic::atomicDec(Policy{}, acc, compare);
}


/*!
 * @brief Atomic bitwise AND equivalent to (*acc) = (*acc) & value
 * This only works with integral data types
 * @param acc Pointer to location of result value
 * @param value Value to bitwise AND with *acc
 * @return Returns value at acc immediately before this operation completed
 */
RAJA_SUPPRESS_HD_WARN
template <typename Policy, typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicAnd(T volatile *acc, T value)
{
  static_assert(std::is_integral<T>::value,
                "atomicAnd can only be used on integral types");
  return RAJA::atomic::atomicAnd(Policy{}, acc, value);
}


/*!
 * @brief Atomic bitwise OR equivalent to (*acc) = (*acc) | value
 * This only works with integral data types
 * @param acc Pointer to location of result value
 * @param value Value to bitwise OR with *acc
 * @return Returns value at acc immediately before this operation completed
 */
RAJA_SUPPRESS_HD_WARN
template <typename Policy, typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicOr(T volatile *acc, T value)
{
  static_assert(std::is_integral<T>::value,
                "atomicOr can only be used on integral types");
  return RAJA::atomic::atomicOr(Policy{}, acc, value);
}


/*!
 * @brief Atomic bitwise XOR equivalent to (*acc) = (*acc) ^ value
 * This only works with integral data types
 * @param acc Pointer to location of result value
 * @param value Value to bitwise XOR with *acc
 * @return Returns value at acc immediately before this operation completed
 */
RAJA_SUPPRESS_HD_WARN
template <typename Policy, typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicXor(T volatile *acc, T value)
{
  static_assert(std::is_integral<T>::value,
                "atomicXor can only be used on integral types");
  return RAJA::atomic::atomicXor(Policy{}, acc, value);
}


/*!
 * @brief Atomic value exchange
 * @param acc Pointer to location to store value
 * @param value Value to exchange with *acc
 * @return Returns value at *acc immediately before this operation completed
 */
RAJA_SUPPRESS_HD_WARN
template <typename Policy, typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicExchange(T volatile *acc, T value)
{
  return RAJA::atomic::atomicExchange(Policy{}, acc, value);
}


/*!
 * @brief Atomic compare and swap
 * @param acc Pointer to location to store value
 * @param value Value to exchange with *acc
 * @param compare Value to compare with *acc
 * @return Returns value at *acc immediately before this operation completed
 */

RAJA_SUPPRESS_HD_WARN
template <typename Policy, typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicCAS(T volatile *acc, T compare, T value)
{
  return RAJA::atomic::atomicCAS(Policy{}, acc, compare, value);
}

/*!
 * \brief Atomic wrapper object
 *
 * Provides an interface akin to that provided by std::atomic, but for an
 * arbitrary memory location.
 *
 * This object provides an OO interface to the global function calls provided
 * as RAJA::atomic::atomicXXX
 *
 * However, the behavior of these operator overloads returns this object,
 * rather than the atomicXXX functions which return the previous value.
 * If your algorithm needs to capture the old value, you must use the functions
 * directly.
 */
template <typename T, typename Policy = auto_atomic>
class AtomicRef
{
public:
  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr explicit AtomicRef(T *value_ptr) : m_value_ptr(value_ptr){};


  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr AtomicRef(AtomicRef<T, Policy> const &c)
      : m_value_ptr(c.m_value_ptr){};


  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr AtomicRef(AtomicRef<T, Policy> &c) : m_value_ptr(c.m_value_ptr){};


  RAJA_INLINE
  RAJA_HOST_DEVICE
  T *getPointer() const { return m_value_ptr; }


  RAJA_INLINE
  RAJA_HOST_DEVICE
  T operator=(T rhs) const
  {
    *m_value_ptr = rhs;
    return rhs;
  }

  RAJA_INLINE
  RAJA_HOST_DEVICE
  T operator++() const { return RAJA::atomic::atomicInc<Policy>(m_value_ptr); }

  RAJA_INLINE
  RAJA_HOST_DEVICE
  T operator++(int)const
  {
    return RAJA::atomic::atomicInc<Policy>(m_value_ptr);
  }

  RAJA_INLINE
  RAJA_HOST_DEVICE
  T operator--() const { return RAJA::atomic::atomicDec<Policy>(m_value_ptr); }

  RAJA_INLINE
  RAJA_HOST_DEVICE
  T operator--(int)const
  {
    return RAJA::atomic::atomicDec<Policy>(m_value_ptr);
  }

  RAJA_INLINE
  RAJA_HOST_DEVICE
  T operator+=(T rhs) const
  {
    return RAJA::atomic::atomicAdd<Policy>(m_value_ptr, rhs);
  }


  RAJA_INLINE
  RAJA_HOST_DEVICE
  T operator-=(T rhs) const
  {
    return RAJA::atomic::atomicSub<Policy>(m_value_ptr, rhs);
  }


  RAJA_INLINE
  RAJA_HOST_DEVICE
  T min(T rhs) const
  {
    return RAJA::atomic::atomicMin<Policy>(m_value_ptr, rhs);
  }


  RAJA_INLINE
  RAJA_HOST_DEVICE
  T max(T rhs) const
  {
    return RAJA::atomic::atomicMax<Policy>(m_value_ptr, rhs);
  }

  RAJA_INLINE
  RAJA_HOST_DEVICE
  T operator&=(T rhs) const
  {
    return RAJA::atomic::atomicAnd<Policy>(m_value_ptr, rhs);
  }

  RAJA_INLINE
  RAJA_HOST_DEVICE
  T operator|=(T rhs) const
  {
    return RAJA::atomic::atomicOr<Policy>(m_value_ptr, rhs);
  }

  RAJA_INLINE
  RAJA_HOST_DEVICE
  T operator^=(T rhs) const
  {
    return RAJA::atomic::atomicXor<Policy>(m_value_ptr, rhs);
  }

  RAJA_INLINE
  RAJA_HOST_DEVICE
  T exchange(T rhs) const
  {
    return RAJA::atomic::atomicExchange<Policy>(m_value_ptr, rhs);
  }

  RAJA_INLINE
  RAJA_HOST_DEVICE
  T CAS(T compare, T rhs) const
  {
    return RAJA::atomic::atomicCAS<Policy>(m_value_ptr, compare, rhs);
  }


private:
  T volatile *m_value_ptr;
};


}  // namespace atomic
}  // namespace RAJA

#endif
