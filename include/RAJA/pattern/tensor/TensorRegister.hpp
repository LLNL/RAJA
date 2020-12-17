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

#ifndef RAJA_pattern_tensor_TensorTensorRegister_HPP
#define RAJA_pattern_tensor_TensorTensorRegister_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#include "camp/camp.hpp"

namespace RAJA
{
  template<typename REGISTER_POLICY,
           typename T,
           typename LAYOUT,
           typename SIZES,
           camp::idx_t SKEW>
  class TensorRegister;

  struct VectorLayout;

  template<typename REGISTER_POLICY, typename T, typename LAYOUT, typename SIZES, camp::idx_t SKEW>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, SKEW>
  operator+(T x, TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, SKEW> const &y){
    using register_t = TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, SKEW>;
    return register_t(x).add(y);
  }

  template<typename REGISTER_POLICY, typename T, typename LAYOUT, typename SIZES, camp::idx_t SKEW>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, SKEW>
  operator-(T x, TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, SKEW> const &y){
    using register_t = TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, SKEW>;
    return register_t(x).subtract(y);
  }

  template<typename REGISTER_POLICY, typename T, typename LAYOUT, typename SIZES, camp::idx_t SKEW>
  TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, SKEW>
  operator*(T x, TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, SKEW> const &y){
    using register_t = TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, SKEW>;
    return register_t(x).multiply(y);
  }

  template<typename REGISTER_POLICY, typename T, typename LAYOUT, typename SIZES, camp::idx_t SKEW>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, SKEW>
  operator/(T x, TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, SKEW> const &y){
    using register_t = TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, SKEW>;
    return register_t(x).divide(y);
  }


namespace internal {

  /*!
   * TensorRegister base class that provides some default behaviors and simplifies
   * the implementation of new register types.
   *
   * This uses CRTP to provide static polymorphism
   */
  template<typename Derived>
  class TensorRegisterBase;

  template<typename REGISTER_POLICY, typename T, typename LAYOUT, typename SIZES, camp::idx_t SKEW>
  class TensorRegisterBase<TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, SKEW>>{
    public:
      using self_type = TensorRegister<REGISTER_POLICY, T, LAYOUT, SIZES, SKEW>;
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
       * Gets the size of the tensor
       * Since this is a vector, just the length of the vector in dim 0
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      constexpr int s_dim_elem(int dim){
        return (dim==0) ? self_type::s_num_elem : 0;
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
       * @brief Broadcast scalar value to first N register elements
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &broadcast_n(element_type const &value, camp::idx_t N){
        for(camp::idx_t i = 0;i < N;++ i){
          getThis()->set(i, value);
        }
        return *getThis();
      }

      /*!
       * @brief Extracts a scalar value and broadcasts to a new register
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type get_and_broadcast(int i) const {
        self_type x;
        x.broadcast(getThis()->get(i));
        return x;
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
        return *getThis();
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
       * @brief Divide n elements of this vector by another vector
       * @param x Vector to divide by
       * @param n Number of elements to divide
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type divide_n(self_type const &b, camp::idx_t n) const {
        self_type q(*getThis());
        for(camp::idx_t i = 0;i < n;++i){
          q.set(i, getThis()->get(i) / b.get(i));
        }
        return q;
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

} //namespace internal


}  // namespace RAJA


// Bring in the register policy file so we get the default register type
// and all of the register traits setup
#include "RAJA/policy/tensor/arch.hpp"


#endif
