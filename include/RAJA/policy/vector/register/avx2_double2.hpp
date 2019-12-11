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

#ifdef __AVX2__

#ifndef RAJA_policy_vector_register_avx2_double2_HPP
#define RAJA_policy_vector_register_avx2_double2_HPP

#include "RAJA/config.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/pattern/register.hpp"

// Include SIMD intrinsics header file
#include <immintrin.h>
#include <cmath>


namespace RAJA
{


  template<>
  class Register<vector_avx2_register, double, 2> :
  public internal::RegisterBase<Register<vector_avx2_register, double, 2>>
{
    public:
      using self_type = Register<vector_avx2_register, double, 2>;
      using element_type = double;
      using register_type = __m128d;

      static constexpr size_t s_num_elem = 2;
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
      void load(element_type const *ptr, size_t stride = 1){
        if(stride == 1){
          m_value = _mm_loadu_pd(ptr);
        }
        else{
          m_value = _mm_i64gather_pd(ptr, _mm_set_epi64x(stride, 0), sizeof(element_type));
        }
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
      void store(element_type *ptr, size_t stride) const{
        if(stride == 1){
          _mm_storeu_pd(ptr, m_value);
        }
        else{
          ptr[0] = m_value[0];
          ptr[stride] = m_value[1];
        }
      }

      /*!
       * @brief Get scalar value from vector register
       * @param i Offset of scalar to get
       * @return Returns scalar value at i
       */
      template<typename IDX>
      constexpr
      RAJA_INLINE
      element_type get(IDX i) const
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

      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      self_type broadcast(element_type const &value){
        return self_type( _mm_set1_pd(value));
      }


      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      void copy(self_type &dst, self_type const &src){
        dst.m_value = src.m_value;
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      self_type add(self_type const &a, self_type const &b){
        return self_type(_mm_add_pd(a.m_value, b.m_value));
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      self_type subtract(self_type const &a, self_type const &b){
        return self_type(_mm_sub_pd(a.m_value, b.m_value));
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      self_type multiply(self_type const &a, self_type const &b){
        return self_type(_mm_mul_pd(a.m_value, b.m_value));
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      self_type divide(self_type const &a, self_type const &b){
        return self_type(_mm_div_pd(a.m_value, b.m_value));
      }


      /*!
       * @brief Sum the elements of this vector
       * @return Sum of the values of the vectors scalar elements
       */
      RAJA_INLINE
      element_type sum() const
      {
        auto hsum = _mm_hadd_pd(m_value, m_value);
        return hsum[0];
      }

      /*!
       * @brief Dot product of two vectors
       * @param x Other vector to dot with this vector
       * @return Value of (*this) dot x
       */
      RAJA_INLINE
      element_type dot(self_type const &x) const
      {
        return self_type(_mm_mul_pd(m_value, x.m_value)).sum();
      }

      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_INLINE
      element_type max() const
      {
        // swap the two lanes
        register_type a = _mm_permute_pd(m_value, 0x01);

        // take the max of each lane (should be same result in each lane)
        register_type b = _mm_max_pd(m_value, a);

        // return the lower lane
        return b[0];
      }

      /*!
       * @brief Returns element-wise largest values
       * @return Vector of the element-wise max values
       */
      RAJA_INLINE
      self_type vmax(self_type a) const
      {
        return self_type(_mm_max_pd(m_value, a.m_value));
      }

      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_INLINE
      element_type min() const
      {
        // swap the two lanes
        register_type a = _mm_permute_pd(m_value, 0x01);

        // take the max of each lane (should be same result in each lane)
        register_type b = _mm_min_pd(m_value, a);

        // return the lower lane
        return b[0];
      }

      /*!
       * @brief Returns element-wise largest values
       * @return Vector of the element-wise max values
       */
      RAJA_INLINE
      self_type vmin(self_type a) const
      {
        return self_type(_mm_min_pd(m_value, a.m_value));
      }
  };



}  // namespace RAJA


#endif

#endif //__AVX2__
