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

#ifndef RAJA_pattern_tensor_TensorRegisterBase_HPP
#define RAJA_pattern_tensor_TensorRegisterBase_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#include "camp/camp.hpp"
#include "RAJA/pattern/tensor/TensorLayout.hpp"
#include "RAJA/pattern/tensor/internal/ET/TensorRef.hpp"

namespace RAJA
{

  struct scalar_register;



namespace internal {

  namespace ET
  {
    class TensorExpressionConcreteBase;
  } // namespace ET


  /*
   * Tensor product helper class.
   *
   * This defines the default product operation between types when using the
   * operator*
   *
   */
  template<typename LHS, typename RHS>
  struct TensorDefaultOperation{

      using multiply_type = decltype(LHS().multiply(RHS()));

      // default multiplication operator
      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      multiply_type multiply(LHS const &lhs, RHS const &rhs)
      {
        return lhs.multiply(rhs);
      }

  };

  class TensorRegisterConcreteBase {};

  /*!
   * TensorRegister base class that provides some default behaviors and simplifies
   * the implementation of new register types.
   *
   * This uses CRTP to provide static polymorphism
   */
  template<typename Derived>
  class TensorRegisterBase;

  template<typename REGISTER_POLICY, typename T, typename LAYOUT, typename camp::idx_t ... SIZES, camp::idx_t ... VAL_SEQ>
  class TensorRegisterBase<TensorRegister<REGISTER_POLICY, T, LAYOUT, camp::idx_seq<SIZES...>, camp::idx_seq<VAL_SEQ...>>> :
    public TensorRegisterConcreteBase
  {
    public:
      using self_type = TensorRegister<REGISTER_POLICY, T, LAYOUT, camp::idx_seq<SIZES...>, camp::idx_seq<VAL_SEQ...>>;
      using element_type = camp::decay<T>;

      static constexpr camp::idx_t s_num_dims = sizeof...(SIZES);

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
       * Gets the default tile of this tensor
       * That tile always start at 0, and extends to the full tile sizes
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      constexpr ET::TensorTile<int, ET::TENSOR_FULL, s_num_dims>
      s_get_default_tile()
      {
        return ET::TensorTile<int, ET::TENSOR_FULL, s_num_dims>{
          {int(SIZES*0)...},
          {int(SIZES)...}
        };
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


      TensorRegisterBase() = default;
      ~TensorRegisterBase() = default;



      TensorRegisterBase(TensorRegisterBase const &) = default;

      RAJA_INLINE
      RAJA_HOST_DEVICE
      TensorRegisterBase(self_type const &){
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
      self_type const &operator=(element_type value)
      {
        getThis()->broadcast(value);
        return *getThis();
      }

      /*!
       * @brief Set entire vector to a single scalar value
       * @param value Value to set all vector elements to
       */
      template<typename T2>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type const &operator=(TensorRegister<scalar_register, T2, ScalarLayout, camp::idx_seq<>, camp::idx_seq<>> const &value)
      {
        getThis()->broadcast(value.get(0));
        return *getThis();
      }

      /*!
       * @brief Assign one register to antoher
       * @param x Vector to copy
       * @return Value of (*this)
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type const &operator=(self_type const &x)
      {
        getThis()->copy(x);
        return *getThis();
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
      template<typename RHS>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      typename TensorDefaultOperation<self_type, RHS>::multiply_type
      operator*(RHS const &rhs) const
      {
        return TensorDefaultOperation<self_type, RHS>::multiply(*getThis(), rhs);
      }

      /*!
       * @brief Multiply a vector with this vector
       * @param x Vector to multiple with this register
       * @return Value of (*this)+x
       */
      template<typename RHS>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator*=(RHS const &rhs)
      {
        *getThis() = TensorDefaultOperation<self_type, RHS>::multiply(*getThis(), rhs);
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
      self_type multiply_add(self_type const &b, self_type const &c) const
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
      self_type multiply_subtract(self_type const &b, self_type const &c) const
      {
        return getThis()->multiply_add(b, -c);
      }

      /*!
       * Multiply this tensor by a scalar value
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type scale(element_type c) const
      {
        return getThis()->multiply(self_type(c));
      }


      /*!
       * In-place add operation
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type &inplace_add(self_type x){
        *getThis() = getThis()->add(x);
        return *getThis();
      }

      /*!
       * In-place sbutract operation
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type &inplace_subtract(self_type x){
        *getThis() = getThis()->subtract(x);
        return *getThis();
      }

      /*!
       * In-place multiply operation
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type &inplace_multiply(self_type x){
        *getThis() = getThis()->multiply(x);
        return *getThis();
      }

      /*!
       * In-place multiply-add operation
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type &inplace_multiply_add(self_type x, self_type y){
        *getThis() = getThis()->multiply_add(x,y);
        return *getThis();
      }

      /*!
       * In-place multiply-subtract operation
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type &inplace_multiply_subtract(self_type x, self_type y){
        *getThis() = getThis()->multiply_subtract(x,y);
        return *getThis();
      }

      /*!
       * In-place divide operation
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type &inplace_divide(self_type x){
        *getThis() = getThis()->divide(x);
        return *getThis();
      }

      /*!
       * In-place scaling operation
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type &inplace_scale(element_type x){
        *getThis() = getThis()->scale(x);
        return *getThis();
      }

  };

} //namespace internal


}  // namespace RAJA



#endif
