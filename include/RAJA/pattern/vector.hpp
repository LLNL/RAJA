/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining vector operations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_vector_HPP
#define RAJA_pattern_vector_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"


// Include SIMD intrinsics header file
#include <immintrin.h>

namespace RAJA
{


/*!
 * \file
 * Vector operation functions in the namespace RAJA

 *
 */

  template<typename T, size_t NUM_ELEM>
  class SimdRegister;

  template<>
  class SimdRegister<double, 4>{
    public:
      using self_type = SimdRegister<double, 4>;
      using element_type = double;

      static constexpr size_t s_num_elem = 4;
      static constexpr size_t s_byte_width = s_num_elem*sizeof(double);
      static constexpr size_t s_bit_width = s_byte_width*8;

      using simd_type = __m256d;

    private:
      simd_type m_value;

    public:

      /*!
       * @brief Default constructor, zeros register contents
       */
      SimdRegister() : m_value(_mm256_setzero_pd()) {
      }

      /*!
       * @brief Copy constructor from underlying simd register
       */
      SimdRegister(simd_type const &c) : m_value(c) {}


      /*!
       * @brief Copy constructor
       */
      SimdRegister(self_type const &c) : m_value(c.m_value) {}

      /*!
       * @brief Get scalar value from vector register
       * @param i Offset of scalar to get
       * @return Returns scalar value at i
       */
      template<typename IDX>
      constexpr
      RAJA_INLINE
      element_type operator[](IDX i) const
      {return m_value[i];}


      /*!
       * @brief Set scalar value in vector register
       * @param i Offset of scalar to set
       * @param value Value of scalar to set
       */
      template<typename IDX>
      RAJA_INLINE
      void set(IDX i, element_type value)
      {m_value[i] = value;}

      /*!
       * @brief Set entire vector to a single scalar value
       * @param value Value to set all vector elements to
       */
      RAJA_INLINE
      self_type const &operator=(element_type value)
      {
        m_value = _mm256_set1_pd(value);
        return *this;
      }

      /*!
       * @brief Assign one register to antoher
       * @param x Vector to copy
       * @return Value of (*this)
       */
      RAJA_INLINE
      self_type const &operator=(self_type const &x)
      {
        m_value = x.m_value;
        return *this;
      }


      /*!
       * @brief Add two vector registers
       * @param x Vector to add to this register
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      self_type operator+(self_type const &x) const
      {
        return self_type(_mm256_add_pd(m_value, x.m_value));
      }

      /*!
       * @brief Add a vector to this vector
       * @param x Vector to add to this register
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      self_type const &operator+=(self_type const &x)
      {
        m_value = _mm256_add_pd(m_value, x.m_value);
        return *this;
      }

      /*!
       * @brief Subtract two vector registers
       * @param x Vector to subctract from this register
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      self_type operator-(self_type const &x) const
      {
        return self_type(_mm256_sub_pd(m_value, x.m_value));
      }

      /*!
       * @brief Subtract a vector from this vector
       * @param x Vector to subtract from this register
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      self_type const &operator-=(self_type const &x)
      {
        m_value = _mm256_sub_pd(m_value, x.m_value);
        return *this;
      }

      /*!
       * @brief Multiply two vector registers, element wise
       * @param x Vector to subctract from this register
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      self_type operator*(self_type const &x) const
      {
        return self_type(_mm256_mul_pd(m_value, x.m_value));
      }

      /*!
       * @brief Multiply a vector with this vector
       * @param x Vector to multiple with this register
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      self_type const &operator*=(self_type const &x)
      {
        m_value = _mm256_mul_pd(m_value, x.m_value);
        return *this;
      }

      /*!
       * @brief Divide two vector registers, element wise
       * @param x Vector to subctract from this register
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      self_type operator/(self_type const &x) const
      {
        return self_type(_mm256_div_pd(m_value, x.m_value));
      }

      /*!
       * @brief Divide this vector by another vector
       * @param x Vector to divide by
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      self_type const &operator/=(self_type const &x)
      {
        m_value = _mm256_div_pd(m_value, x.m_value);
        return *this;
      }

      /*!
       * @brief Sum the elements of this vector
       * @return Sum of the values of the vectors scalar elements
       */
      RAJA_INLINE
      element_type sum() const
      {
        auto hsum = _mm256_hadd_pd(m_value, m_value);
        return hsum[0] + hsum[2];
      }

      /*!
       * @brief Dot product of two vectors
       * @param x Other vector to dot with this vector
       * @return Value of (*this) dot x
       */
      RAJA_INLINE
      element_type dot(self_type const &x) const
      {
        return self_type(_mm256_mul_pd(m_value, x.m_value)).sum();
      }
  };


  /**
   * A specialization for a single element SIMD register.
   * We will implement this as a scalar value, and let the compiler use
   * whatever registers it deems appropriate.
   */
  template<typename T>
  class SimdRegister<T, 1>{
    public:
      using self_type = SimdRegister<T, 1>;
      using element_type = T;

      static constexpr size_t s_num_elem = 1;
      static constexpr size_t s_byte_width = sizeof(T);
      static constexpr size_t s_bit_width = s_byte_width*8;

    private:
      T m_value;

    public:

      /*!
       * @brief Default constructor, zeros register contents
       */
      SimdRegister() : m_value(0) {
      }

      /*!
       * @brief Copy constructor from underlying simd register
       */
      SimdRegister(T const &c) : m_value(c) {}


      /*!
       * @brief Copy constructor
       */
      SimdRegister(self_type const &c) : m_value(c.m_value) {}

      /*!
       * @brief Get scalar value from vector register
       * @param i Offset of scalar to get
       * @return Returns scalar value at i
       */
      template<typename IDX>
      constexpr
      RAJA_INLINE
      element_type operator[](IDX) const
      {return m_value;}


      /*!
       * @brief Set scalar value in vector register
       * @param i Offset of scalar to set
       * @param value Value of scalar to set
       */
      template<typename IDX>
      RAJA_INLINE
      void set(IDX , element_type value)
      {m_value = value;}

      /*!
       * @brief Set entire vector to a single scalar value
       * @param value Value to set all vector elements to
       */
      RAJA_INLINE
      self_type const &operator=(element_type value)
      {
        m_value = value;
        return *this;
      }

      /*!
       * @brief Assign one register to antoher
       * @param x Vector to copy
       * @return Value of (*this)
       */
      RAJA_INLINE
      self_type const &operator=(self_type const &x)
      {
        m_value = x.m_value;
        return *this;
      }


      /*!
       * @brief Add two vector registers
       * @param x Vector to add to this register
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      self_type operator+(self_type const &x) const
      {
        return self_type(m_value + x.m_value);
      }

      /*!
       * @brief Add a vector to this vector
       * @param x Vector to add to this register
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      self_type const &operator+=(self_type const &x)
      {
        m_value = m_value + x.m_value;
        return *this;
      }

      /*!
       * @brief Subtract two vector registers
       * @param x Vector to subctract from this register
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      self_type operator-(self_type const &x) const
      {
        return self_type(m_value - x.m_value);
      }

      /*!
       * @brief Subtract a vector from this vector
       * @param x Vector to subtract from this register
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      self_type const &operator-=(self_type const &x)
      {
        m_value = m_value - x.m_value;
        return *this;
      }

      /*!
       * @brief Multiply two vector registers, element wise
       * @param x Vector to subctract from this register
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      self_type operator*(self_type const &x) const
      {
        return self_type(m_value * x.m_value);
      }

      /*!
       * @brief Multiply a vector with this vector
       * @param x Vector to multiple with this register
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      self_type const &operator*=(self_type const &x)
      {
        m_value = m_value * x.m_value;
        return *this;
      }

      /*!
       * @brief Divide two vector registers, element wise
       * @param x Vector to subctract from this register
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      self_type operator/(self_type const &x) const
      {
        return self_type(m_value / x.m_value);
      }

      /*!
       * @brief Divide this vector by another vector
       * @param x Vector to divide by
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      self_type const &operator/=(self_type const &x)
      {
        m_value = m_value / x.m_value;
        return *this;
      }

      /*!
       * @brief Sum the elements of this vector
       * @return Sum of the values of the vectors scalar elements
       */
      RAJA_INLINE
      element_type sum() const
      {
        return m_value;
      }

      /*!
       * @brief Dot product of two vectors
       * @param x Other vector to dot with this vector
       * @return Value of (*this) dot x
       */
      RAJA_INLINE
      element_type dot(self_type const &x) const
      {
        return m_value*x.m_value;
      }
  };



//  template<typename REGISTER, size_t BIT_WIDTH>
//  struct FixedVector {
//    public:
//      using self_type = FixedVector<REGISTER, BIT_WIDTH>;
//      using element_type = typename REGISTER::element_type;
//
//      static constexpr size_t s_bit_width = 256;
//      static constexpr size_t s_byte_width = s_bit_width/8;
//      static constexpr size_t s_num_elem = s_byte_width / sizeof(double);
//
//      using register_type = REGISTER;
//
//    private:
//      register_type m_values[s_num_registers];
//
//    public:
//  };




}  // namespace RAJA

#endif
