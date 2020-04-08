/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining SIMD/SIMT register operations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_vector_registerbase_HPP
#define RAJA_pattern_vector_registerbase_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

namespace RAJA
{

  namespace internal {
  /*!
   * Register base class that provides some default behaviors and simplifies
   * the implementation of new register types.
   *
   * This uses CRTP to provide static polymorphism
   */
  template<typename Derived>
  class RegisterBase;

  template<typename REGISTER_POLICY, typename T>
  class RegisterBase<Register<REGISTER_POLICY, T>>{
    public:
      using self_type = Register<REGISTER_POLICY, T>;
      using element_type = camp::decay<T>;



    private:

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type *getThis(){
        return static_cast<self_type *>(this);
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      self_type const *getThis() const{
        return static_cast<self_type const *>(this);
      }

    public:

      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      constexpr
      bool is_root() {
        return true;
      }

      /*!
       * @brief convenience routine to allow Vector classes to use
       * camp::sink() across a variety of register types, and use things like
       * ternary operators
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      bool sink() const{
        return false;
      }

      /*!
       * @brief Set entire vector to a single scalar value
       * @param value Value to set all vector elements to
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator=(element_type value)
      {
        getThis()->broadcast(value);
        return *this;
      }

      /*!
       * @brief Assign one register to antoher
       * @param x Vector to copy
       * @return Value of (*this)
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator=(self_type const &x)
      {
        getThis()->copy(x);
        return *this;
      }

      /*!
       * @brief Get scalar value from vector register
       * @param i Offset of scalar to get
       * @return Returns scalar value at i
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      element_type operator[](camp::idx_t i) const
      {
        return getThis()->get(i);
      }

      /*!
       * @brief Add two vector registers
       * @param x Vector to add to this register
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type operator+(self_type const &x) const
      {
        return getThis()->add(x);
      }


      /*!
       * @brief Add a vector to this vector
       * @param x Vector to add to this register
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator+=(self_type const &x)
      {
        *getThis() = getThis()->add(x);
        return *getThis();
      }

      /*!
       * @brief Negate the value of this vector
       * @return Value of -(*this)
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type operator-() const
      {
        return self_type(0).subtract(*getThis());
      }

      /*!
       * @brief Subtract two vector registers
       * @param x Vector to subctract from this register
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type operator-(self_type const &x) const
      {
        return getThis()->subtract(x);
      }

      /*!
       * @brief Subtract a vector from this vector
       * @param x Vector to subtract from this register
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator-=(self_type const &x)
      {
        *getThis() = getThis()->subtract(x);
        return *getThis();
      }

      /*!
       * @brief Multiply two vector registers, element wise
       * @param x Vector to subctract from this register
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type operator*(self_type const &x) const
      {
        return getThis()->multiply(x);
      }

      /*!
       * @brief Multiply a vector with this vector
       * @param x Vector to multiple with this register
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator*=(self_type const &x)
      {
        *getThis() = getThis()->multiply(x);
        return *getThis();
      }

      /*!
       * @brief Divide two vector registers, element wise
       * @param x Vector to subctract from this register
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type operator/(self_type const &x) const
      {
        return getThis()->divide(x);
      }

      /*!
       * @brief Divide this vector by another vector
       * @param x Vector to divide by
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator/=(self_type const &x)
      {
        *getThis() = getThis()->divide(x);
        return *getThis();
      }


      /*!
       * @brief Dot product of two vectors
       * @param x Other vector to dot with this vector
       * @return Value of (*this) dot x
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      element_type dot(self_type const &x) const
      {
        return getThis()->multiply(x).sum();
      }

      /*!
       * @brief Fused multiply add: fma(b, c) = (*this)*b+c
       *
       * Derived types can override this to implement intrinsic FMA's
       *
       * @param b Second product operand
       * @param c Sum operand
       * @return Value of (*this)*b+c
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type fused_multiply_add(self_type const &b, self_type const &c) const
      {
        return (self_type(*getThis()) * self_type(b)) + self_type(c);
      }

      /*!
       * @brief Fused multiply subtract: fms(b, c) = (*this)*b-c
       *
       * Derived types can override this to implement intrinsic FMS's
       *
       * @param b Second product operand
       * @param c Subtraction operand
       * @return Value of (*this)*b-c
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type fused_multiply_subtract(self_type const &b, self_type const &c) const
      {
        return getThis()->fused_multiply_add(b, -c);
      }

  };

  }
}  // namespace RAJA



#endif
