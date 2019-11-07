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

  template<typename REGISTER_INDEX, typename POINTER_TYPE, bool STRIDE_ONE>
  class VectorRef {
    public:
      using self_type = VectorRef<REGISTER_INDEX, POINTER_TYPE, STRIDE_ONE>;
      using register_index_type = REGISTER_INDEX;
      using register_type = typename register_index_type::register_type;
      using element_type = typename register_type::element_type;
      using pointer_type = POINTER_TYPE;

    private:
      register_index_type m_index;
      pointer_type m_data;
      size_t m_stride;

    public:


      /*!
       * @brief Default constructor, zeros register contents
       */
      RAJA_INLINE
      VectorRef() : m_index(), m_data() {};

      /*!
       * @brief Stride-1 constructor
       */
      RAJA_INLINE
      VectorRef(register_index_type index, pointer_type pointer) :
      m_index(index),
      m_data(pointer),
      m_stride(1)
          {}


      /*!
       * @brief Strided constructor
       */
      RAJA_INLINE
      VectorRef(register_index_type index, pointer_type pointer, size_t stride) :
      m_index(index),
      m_data(pointer),
      m_stride(stride)
          {}

      /*!
       * @brief Copy constructor
       */
      RAJA_INLINE
      VectorRef(self_type const &c) :
      m_index(c.m_index),
      m_data(c.m_data)
          {}


      /*!
       * @brief Set entire vector to a single scalar value
       * @param value Value to set all vector elements to
       */
      RAJA_INLINE
      void store(register_type value) const
      {
        if(STRIDE_ONE){
          value.store(m_data+*m_index);
        }
        else{
          value.store(m_data+*m_index, m_stride);
        }
      }

      /*!
       * @brief Set entire vector to a single scalar value
       * @param value Value to set all vector elements to
       */
      RAJA_INLINE
      register_type load() const
      {
        register_type value;
        if(STRIDE_ONE){
          value.load(m_data+*m_index);
        }
        else{
          value.load(m_data+*m_index, m_stride);
        }
        return value;
      }

      RAJA_INLINE
      operator register_type() const {
        return load();
      }


      /*!
       * @brief Set entire vector to a single scalar value
       * @param value Value to set all vector elements to
       */
      RAJA_INLINE
      self_type const &operator=(register_type value)
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
      constexpr
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
      RAJA_INLINE
      void set(IDX i, element_type value)
      {
        register_type x = load();
        x[i] = value;
        store(x);
      }

      /*!
       * @brief Set entire vector to a single scalar value
       * @param value Value to set all vector elements to
       */
      RAJA_INLINE
      self_type const &operator=(element_type value)
      {
        register_type x = value;
        store(x);
        return *this;
      }


      /*!
       * @brief Add two vector registers
       * @param x Vector to add to this register
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      register_type operator+(register_type const &x) const
      {
        return load() + x;
      }

      /*!
       * @brief Add a vector to this vector
       * @param x Vector to add to this register
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      self_type const &operator+=(register_type const &x)
      {
        store(load() + x);
        return *this;
      }

      /*!
       * @brief Subtract two vector registers
       * @param x Vector to subctract from this register
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      register_type operator-(register_type const &x) const
      {
        return load() - x;
      }

      /*!
       * @brief Subtract a vector from this vector
       * @param x Vector to subtract from this register
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      self_type const &operator-=(register_type const &x)
      {
        store(load() - x);
        return *this;
      }

      /*!
       * @brief Multiply two vector registers, element wise
       * @param x Vector to subctract from this register
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      register_type operator*(register_type const &x) const
      {
        return load() * x;
      }

      /*!
       * @brief Multiply a vector with this vector
       * @param x Vector to multiple with this register
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      self_type const &operator*=(register_type const &x)
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
      register_type operator/(register_type const &x) const
      {
        return load() / x;
      }

      /*!
       * @brief Divide this vector by another vector
       * @param x Vector to divide by
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      self_type const &operator/=(register_type const &x)
      {
        store(load() / x);
        return *this;
      }

      /*!
       * @brief Sum the elements of this vector
       * @return Sum of the values of the vectors scalar elements
       */
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
      RAJA_INLINE
      element_type dot(register_type const &x) const
      {
        return load().dot(x);
      }

      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_INLINE
      element_type max() const
      {
        return load().max();
      }

      /*!
       * @brief Returns element-wise largest values
       * @return Vector of the element-wise max values
       */
      RAJA_INLINE
      register_type vmax(register_type a) const
      {
        return load().vmax(a);
      }

      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_INLINE
      element_type min() const
      {
        return load().min();
      }

      /*!
       * @brief Returns element-wise largest values
       * @return Vector of the element-wise max values
       */
      RAJA_INLINE
      register_type vmin(register_type a) const
      {
        return load().vmin(a);
      }

  };

}  // namespace RAJA


#endif
