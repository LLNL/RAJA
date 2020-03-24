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

#ifndef RAJA_pattern_vector_vectorref_HPP
#define RAJA_pattern_vector_vectorref_HPP

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

  template<typename VECTOR_TYPE, typename INDEX_TYPE,
           typename POINTER_TYPE, bool STRIDE_ONE>
  class VectorRef {
    public:
      using self_type =
          VectorRef<VECTOR_TYPE, INDEX_TYPE, POINTER_TYPE, STRIDE_ONE>;

      using vector_type = VECTOR_TYPE;
      using index_type = INDEX_TYPE;
      using pointer_type = POINTER_TYPE;

      using element_type = typename vector_type::element_type;


    private:
      index_type m_linear_index;
      index_type m_length;
      pointer_type m_data;
      index_type m_stride;

    public:


      /*!
       * @brief Default constructor, zeros register contents
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      VectorRef() : m_linear_index(0), m_length(0), m_data(), m_stride(0) {};

      /*!
       * @brief Constructor
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      VectorRef(index_type lin_index, index_type length, pointer_type pointer, index_type stride) :
      m_linear_index(lin_index),
      m_length(length),
      m_data(pointer),
      m_stride(stride)
          {}


      /*!
       * @brief Copy constructor
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      VectorRef(self_type const &c) :
      m_linear_index(c.m_linear_index),
      m_length(c.m_length),
      m_data(c.m_data),
      m_stride(c.m_stride)
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
      constexpr
      element_type *get_pointer() const
      {
        return &m_data[m_linear_index];
      }

      /*!
       * @brief Set entire vector to a single scalar value
       * @param value Value to set all vector elements to
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      void store(vector_type value) const
      {
        if(STRIDE_ONE){
          value.store(m_data+m_linear_index);
        }
        else{
          value.store(m_data+m_linear_index, m_stride);
        }
      }

      /*!
       * @brief Set entire vector to a single scalar value
       * @param value Value to set all vector elements to
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      vector_type load() const
      {
        vector_type value;
        if(STRIDE_ONE){
          value.load(m_data+m_linear_index, 1, m_length);
        }
        else{
          value.load(m_data+m_linear_index, m_stride, m_length);
        }
        return value;
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      index_type size() const
      {
        return m_length;
      }

      /*!
       * @brief Automatic conversion to the underlying vector_type.
       *
       * This allows the use of a VectorRef in an expression, and lets the
       * compiler automatically convert a VectorRef into a load().
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      operator vector_type() const {
        return load();
      }


      /*!
       * @brief Set entire vector to a single scalar value
       * @param value Value to set all vector elements to
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator=(vector_type value)
      {
        store(value);
        return *this;
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
        return load()[i];
      }


      /*!
       * @brief Set scalar value in vector register
       * @param i Offset of scalar to set
       * @param value Value of scalar to set
       */
      template<typename IDX>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      void set(IDX i, element_type value)
      {
        vector_type x = load();
        x[i] = value;
        store(x);
      }

      /*!
       * @brief Set entire vector to a single scalar value
       * @param value Value to set all vector elements to
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type const &operator=(element_type value)
      {
        vector_type x = value;
        store(x);
        return *this;
      }


      /*!
       * @brief Add two vector registers
       * @param x Vector to add to this register
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      vector_type operator+(vector_type const &x) const
      {
        return load() + x;
      }

      /*!
       * @brief Add a vector to this vector
       * @param x Vector to add to this register
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator+=(vector_type const &x)
      {
        store(load() + x);
        return *this;
      }

      /*!
       * @brief Add a product of two vectors, resulting in an FMA
       * @param x Vector to add to this register
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator+=(VectorProductRef<vector_type> const &x)
      {
        store(load() + x);
        return *this;
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
        return load() - x;
      }

      /*!
       * @brief Subtract a vector from this vector
       * @param x Vector to subtract from this register
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator-=(vector_type const &x)
      {
        store(load() - x);
        return *this;
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
        return VectorProductRef<vector_type>(load(), x);
      }

      /*!
       * @brief Multiply a vector with this vector
       * @param x Vector to multiple with this register
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator*=(vector_type const &x)
      {
        store(load() * x);
        return *this;
      }

      /*!
       * @brief Divide two vector registers, element wise
       * @param x Vector to subctract from this register
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      vector_type operator/(vector_type const &x) const
      {
        return load() / x;
      }

      /*!
       * @brief Divide this vector by another vector
       * @param x Vector to divide by
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator/=(vector_type const &x)
      {
        store(load() / x);
        return *this;
      }

      /*!
       * @brief Sum the elements of this vector
       * @return Sum of the values of the vectors scalar elements
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type sum() const
      {
        return load().sum();
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
        return load().dot(x);
      }

      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type max() const
      {
        return load().max();
      }

      /*!
       * @brief Returns element-wise largest values
       * @return Vector of the element-wise max values
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      vector_type vmax(vector_type a) const
      {
        return load().vmax(a);
      }

      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type min() const
      {
        return load().min();
      }

      /*!
       * @brief Returns element-wise largest values
       * @return Vector of the element-wise max values
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      vector_type vmin(vector_type a) const
      {
        return load().vmin(a);
      }

  };



  template<typename VECTOR_TYPE, typename INDEX_TYPE, typename POINTER_TYPE, bool STRIDE_ONE>
  VECTOR_TYPE
  operator+(typename VECTOR_TYPE::element_type x, VectorRef<VECTOR_TYPE, INDEX_TYPE, POINTER_TYPE, STRIDE_ONE> const &y){
    return VECTOR_TYPE(x) + y.load();
  }

  template<typename VECTOR_TYPE, typename INDEX_TYPE, typename POINTER_TYPE, bool STRIDE_ONE>
  VECTOR_TYPE
  operator-(typename VECTOR_TYPE::element_type x, VectorRef<VECTOR_TYPE, INDEX_TYPE, POINTER_TYPE, STRIDE_ONE> const &y){
    return VECTOR_TYPE(x) - y.load();
  }

  template<typename VECTOR_TYPE, typename INDEX_TYPE, typename POINTER_TYPE, bool STRIDE_ONE>
  VectorProductRef<VECTOR_TYPE>
  operator*(typename VECTOR_TYPE::element_type x, VectorRef<VECTOR_TYPE, INDEX_TYPE, POINTER_TYPE, STRIDE_ONE> const &y){
    return VectorProductRef<VECTOR_TYPE>(VECTOR_TYPE(x), y.load());
  }

  template<typename VECTOR_TYPE, typename INDEX_TYPE, typename POINTER_TYPE, bool STRIDE_ONE>
  VECTOR_TYPE
  operator/(typename VECTOR_TYPE::element_type x, VectorRef<VECTOR_TYPE, INDEX_TYPE, POINTER_TYPE, STRIDE_ONE> const &y){
    return VECTOR_TYPE(x) / y.load();
  }


}  // namespace RAJA


#endif
