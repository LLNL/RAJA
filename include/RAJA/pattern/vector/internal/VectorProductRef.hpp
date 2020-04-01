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

#ifndef RAJA_pattern_vector_vectorproductref_HPP
#define RAJA_pattern_vector_vectorproductref_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#include <array>

namespace RAJA
{


/*!
 * \file
 * Vector operation functions in the namespace RAJA

 *
 */

  template<typename VECTOR_TYPE>
  class VectorProductRef {
    public:
      using self_type =
          VectorProductRef<VECTOR_TYPE>;

      using vector_type = VECTOR_TYPE;
      using element_type = typename vector_type::element_type;

    private:
      vector_type m_vector_a;
      vector_type m_vector_b;

    public:


      /*!
       * @brief Default constructor, zeros register contents
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      VectorProductRef() {};

      /*!
       * @brief Constructor
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      VectorProductRef(vector_type const &a, vector_type const &b) :
      m_vector_a(a),
      m_vector_b(b)
          {}


      /*!
       * @brief Copy constructor
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      VectorProductRef(self_type const &c) :
      m_vector_a(c.m_vector_a),
      m_vector_b(c.m_vector_b)
          {}


      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      constexpr
      bool is_root() {
        return vector_type::is_root();
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      vector_type get_result() const
      {
        vector_type result = m_vector_a;
        result *= m_vector_b;
        return result;
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      vector_type const &get_left() const
      {
        return m_vector_a;
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      vector_type const &get_right() const
      {
        return m_vector_b;
      }


      /*!
       * @brief Automatic conversion to the underlying vector_type.
       *
       * This will evaluate the underlying operation on the two vectors
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      operator vector_type() const {
        return get_result();
      }



      /*!
       * @brief Get scalar value from vector register
       * @param i Offset of scalar to get
       * @return Returns scalar value at i
       */
      template<typename IDX>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type operator[](IDX i) const
      {
        return get_result()[i];
      }




      /*!
       * @brief Add this product to another vector with an FMA.
       * @param x Vector to add to this product
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      vector_type operator+(vector_type const &x) const
      {
        return m_vector_a.fused_multiply_add(m_vector_b, x);
      }



      /*!
       * @brief Subtract two vector registers
       * @param x Vector to subctract from this register
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      vector_type operator-(vector_type const &x) const
      {
        return m_vector_a.fused_multiply_subtract(m_vector_b, x);
      }



      /*!
       * @brief Multiply two vector registers, element wise
       * @param x Vector to subctract from this register
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      VectorProductRef<vector_type> operator*(vector_type const &x) const
      {
        return VectorProductRef<vector_type>(get_result(), x);
      }



      /*!
       * @brief Divide two vector registers, element wise
       * @param x Vector to subctract from this register
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      vector_type operator/(vector_type const &x) const
      {
        return get_result() / x;
      }



      /*!
       * @brief Sum the elements of this vector
       * @return Sum of the values of the vectors scalar elements
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type sum() const
      {
        return get_result().sum();
      }

      /*!
       * @brief Dot product of two vectors
       * @param x Other vector to dot with this vector
       * @return Value of (*this) dot x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type dot(vector_type const &x) const
      {
        return get_result().dot(x);
      }

      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type max() const
      {
        return get_result().max();
      }

      /*!
       * @brief Returns element-wise largest values
       * @return Vector of the element-wise max values
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      vector_type vmax(vector_type a) const
      {
        return get_result().vmax(a);
      }

      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type min() const
      {
        return get_result().min();
      }

      /*!
       * @brief Returns element-wise largest values
       * @return Vector of the element-wise max values
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      vector_type vmin(vector_type a) const
      {
        return get_result().vmin(a);
      }

  };



  template<typename ST, typename VECTOR_TYPE>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  VECTOR_TYPE
  operator+(ST x, VectorProductRef<VECTOR_TYPE> const &y){
    return y.get_left().fused_multiply_add(y.get_right(), x);
  }

  template<typename ST, typename VECTOR_TYPE>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  VECTOR_TYPE
  operator-(ST x, VectorProductRef<VECTOR_TYPE> const &y){
    y.get_left().fused_multiply_subtract(y.get_right(), x);
  }

  template<typename ST, typename VECTOR_TYPE>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  VectorProductRef<VECTOR_TYPE>
  operator*(ST x, VectorProductRef<VECTOR_TYPE> const &y){
    return VectorProductRef<VECTOR_TYPE>(VECTOR_TYPE(x), y.get_result());
  }

  template<typename ST, typename VECTOR_TYPE>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  VECTOR_TYPE
  operator/(ST x, VectorProductRef<VECTOR_TYPE> const &y){
    return VECTOR_TYPE(x) / y.get_result();
  }


}  // namespace RAJA


#endif
