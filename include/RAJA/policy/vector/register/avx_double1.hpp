/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining a SIMD register abstraction.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifdef __AVX__

#ifndef RAJA_policy_vector_register_avx_double1_HPP
#define RAJA_policy_vector_register_avx_double1_HPP

#include "RAJA/config.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/pattern/register.hpp"

// Include SIMD intrinsics header file
#include <immintrin.h>
#include <cmath>


namespace RAJA
{


  template<>
  class Register<vector_avx_register, double, 1> :
    public internal::RegisterBase<Register<vector_avx_register, double, 1>>
  {
    public:
      using self_type = Register<vector_avx_register, double, 1>;
      using element_type = double;
      using register_type = __m128d;

      static constexpr size_t s_num_elem = 1;
      static constexpr size_t s_byte_width = s_num_elem*sizeof(double);
      static constexpr size_t s_bit_width = s_byte_width*8;

    private:
      register_type m_value;

    public:

      /*!
       * @brief Default constructor, zeros register contents
       */
      RAJA_INLINE
      Register() : m_value(_mm_setzero_pd()) {
      }

      /*!
       * @brief Copy constructor from underlying simd register
       */
      RAJA_INLINE
      constexpr
      explicit Register(register_type const &c) : m_value(c) {}


      /*!
       * @brief Copy constructor
       */
      RAJA_INLINE
      constexpr
      Register(self_type const &c) : m_value(c.m_value) {}

      /*!
       * @brief Construct from scalar.
       * Sets all elements to same value (broadcast).
       */
      RAJA_INLINE
      Register(element_type const &c) : m_value(_mm_set1_pd(c)) {}


      /*!
       * @brief Strided load operation, when scalars are located in memory
       * locations ptr, ptr+stride
       *
       *
       * Note: this could be done with "gather" instructions if they are
       * available. (like in avx2, but not in avx)
       */
      RAJA_INLINE
      self_type &load(element_type const *ptr, size_t  = 1){
        m_value[0] = ptr[0];
        return *this;
      }


      /*!
       * @brief Strided store operation, where scalars are stored in memory
       * locations ptr, ptr+stride, ptr+2*stride, etc.
       *
       *
       * Note: this could be done with "scatter" instructions if they are
       * available.
       */
      RAJA_INLINE
      self_type const &store(element_type *ptr, size_t = 1) const{
        ptr[0] = m_value[0];
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
      element_type get(IDX ) const
      {return m_value[0];}


      /*!
       * @brief Set scalar value in vector register
       * @param i Offset of scalar to set
       * @param value Value of scalar to set
       */
      template<typename IDX>
      RAJA_INLINE
      self_type &set(IDX , element_type value)
      {
        m_value[0] = value;
        return *this;
      }


      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &broadcast(element_type const &b){
        m_value[0] = b;
        return *this;
      }


      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &copy(self_type const &src){
        m_value = src.m_value;
        return *this;
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type add(self_type const &b) const {
        return self_type(_mm_add_sd(m_value, b.m_value));
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type subtract(self_type const &b) const {
        return self_type(_mm_sub_sd(m_value, b.m_value));
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type multiply(self_type const &b) const {
        return self_type(_mm_mul_sd(m_value, b.m_value));
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type divide(self_type const &b) const {
        return self_type(_mm_div_sd(m_value, b.m_value));
      }

      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type fused_multiply_add(self_type const &b, self_type const &c) const
      {
        return self_type(_mm_fmadd_sd(m_value, b.m_value, c.m_value));
      }


      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type fused_multiply_subtract(self_type const &b, self_type const &c) const
      {
        return self_type(_mm_fmsub_sd(m_value, b.m_value, c.m_value));
      }

      /*!
       * @brief Sum the elements of this vector
       * @return Sum of the values of the vectors scalar elements
       */
      RAJA_INLINE
      element_type sum() const
      {
        return m_value[0];
      }

      /*!
       * @brief Dot product of two vectors
       * @param x Other vector to dot with this vector
       * @return Value of (*this) dot x
       */
      RAJA_INLINE
      element_type dot(self_type const &x) const
      {
        return _mm_mul_sd(m_value, x.m_value)[0];
      }

      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_INLINE
      element_type max() const
      {
        return m_value[0];
      }

      /*!
       * @brief Returns element-wise largest values
       * @return Vector of the element-wise max values
       */
      RAJA_INLINE
      self_type vmax(self_type a) const
      {
        return RAJA::max(m_value[0], a.m_value[0]);
      }

      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_INLINE
      element_type min() const
      {
        return m_value[0];
      }

      /*!
       * @brief Returns element-wise largest values
       * @return Vector of the element-wise max values
       */
      RAJA_INLINE
      self_type vmin(self_type a) const
      {
        return RAJA::min(m_value[0], a.m_value[0]);
      }
  };



}  // namespace RAJA


#endif

#endif //__AVX__

