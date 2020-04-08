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

#ifndef RAJA_policy_vector_register_avx2_int64_HPP
#define RAJA_policy_vector_register_avx2_int64_HPP

#include "RAJA/config.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/pattern/vector.hpp"

// Include SIMD intrinsics header file
#include <immintrin.h>
#include <cmath>


namespace RAJA
{

  template<>
  class Register<avx2_register, long> :
    public internal::RegisterBase<Register<avx2_register, long>>
  {
    public:
      using register_policy = avx2_register;
      using self_type = Register<avx2_register, long>;
      using element_type = long;
      using register_type = __m256i;


    private:
      register_type m_value;

      RAJA_INLINE
      __m256i createMask(camp::idx_t N) const {
        // Generate a mask
        return  _mm256_set_epi64x(
            N >= 4 ? -1 : 0,
            N >= 3 ? -1 : 0,
            N >= 2 ? -1 : 0,
            N >= 1 ? -1 : 0);
      }

      RAJA_INLINE
      __m256i createStridedOffsets(camp::idx_t stride) const {
        // Generate a strided offset list
        return  _mm256_set_epi64x(3*stride, 2*stride, stride, 0);
      }

      /*
       * Use the packed-double permute function because there isn't one
       * specifically for int64
       *
       * Just adds a bunch of casting, should be same cost
       */
      template<int perm>
      RAJA_INLINE
      __m256i permute(__m256i x) const {
        return _mm256_castpd_si256(
            _mm256_permute_pd(_mm256_castsi256_pd(x), perm));
      }

    public:

      static constexpr camp::idx_t s_num_elem = 4;


      /*!
       * @brief Default constructor, zeros register contents
       */
      RAJA_INLINE
      Register() : m_value(_mm256_setzero_si256()) {
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
      Register(element_type const &c) : m_value(_mm256_set1_epi64x(c)) {}


      /*!
       * @brief Strided load constructor, when scalars are located in memory
       * locations ptr, ptr+stride, ptr+2*stride, etc.
       *
       *
       * Note: this could be done with "gather" instructions if they are
       * available. (like in avx2, but not in avx)
       */
      RAJA_INLINE
      self_type &load(element_type const *ptr, camp::idx_t stride = 1, camp::idx_t N = 4){
        // no elements
        if(N <= 0){
          m_value = _mm256_setzero_si256();
        }
        // Full vector width uses regular load/gather instruction
        if(N == 4){

          // Packed Load
          if(stride == 1){
            m_value = _mm256_loadu_si256(reinterpret_cast<__m256i const *>(ptr));
          }

          // Gather
          else{
            m_value = _mm256_i64gather_epi64(reinterpret_cast<long long const *>(ptr),
                                          createStridedOffsets(stride),
                                          sizeof(element_type));
          }
        }

        // Not-full vector (1,2 or 3 doubles) uses a masked load/gather
        else {

          // Masked Packed Load
          if(stride == 1){
            m_value = _mm256_castpd_si256(
                _mm256_maskload_pd(reinterpret_cast<double const *>(ptr), createMask(N))
            );
          }

          // Masked Gather
          else{
            m_value = _mm256_mask_i64gather_epi64(_mm256_set1_epi64x(0),
                                          reinterpret_cast<long long const *>(ptr),
                                          createStridedOffsets(stride),
                                          createMask(N),
                                          sizeof(element_type));
          }
        }
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
      self_type const &store(element_type *ptr, camp::idx_t stride = 1, camp::idx_t N = 4) const{
        // Is this a packed store?
        if(stride == 1){
          // Is it full-width?
          if(N == 4){
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), m_value);
          }
          // Need to do a masked store
          else{
            _mm256_maskstore_epi64(reinterpret_cast<long long*>(ptr), createMask(N), m_value);
          }

        }

        // Scatter operation:  AVX2 doesn't have a scatter, so it's manual
        else{
          for(camp::idx_t i = 0;i < N;++ i){
            ptr[i*stride] = get(i);
          }
        }
        return *this;
      }

      /*!
       * @brief Get scalar value from vector register
       * @param i Offset of scalar to get
       * @return Returns scalar value at i
       */
      RAJA_INLINE
      element_type get(camp::idx_t i) const
      {
        // got to be a nicer way to do this!?!?
        switch(i){
          case 0: return _mm256_extract_epi64(m_value, 0);
          case 1: return _mm256_extract_epi64(m_value, 1);
          case 2: return _mm256_extract_epi64(m_value, 2);
          case 3: return _mm256_extract_epi64(m_value, 3);
        }
        return 0;
      }


      /*!
       * @brief Set scalar value in vector register
       * @param i Offset of scalar to set
       * @param value Value of scalar to set
       */
      RAJA_INLINE
      self_type &set(camp::idx_t i, element_type value)
      {
        // got to be a nicer way to do this!?!?
        switch(i){
          case 0: m_value = _mm256_insert_epi64(m_value, value, 0); break;
          case 1: m_value = _mm256_insert_epi64(m_value, value, 1); break;
          case 2: m_value = _mm256_insert_epi64(m_value, value, 2); break;
          case 3: m_value = _mm256_insert_epi64(m_value, value, 3); break;
        }

        return *this;
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &broadcast(element_type const &value){
        m_value =  _mm256_set1_epi64x(value);
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
        return self_type(_mm256_add_epi64(m_value, b.m_value));
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type subtract(self_type const &b) const {
        return self_type(_mm256_sub_epi64(m_value, b.m_value));
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type multiply(self_type const &b) const {
        // AVX2 does not supply an long multiply, so do it manually
        return self_type(_mm256_set_epi64x(
            get(3)*b.get(3),
            get(2)*b.get(2),
            get(1)*b.get(1),
            get(0)*b.get(0)
            ));
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type divide(self_type const &b, camp::idx_t N = 4) const {
        // AVX2 does not supply an integer divide, so do it manually
        return self_type(_mm256_set_epi64x(
            N >= 4 ? get(3)/b.get(3) : 0,
            N >= 3 ? get(2)/b.get(2) : 0,
            N >= 2 ? get(1)/b.get(1) : 0,
            N >= 1 ? get(0)/b.get(0) : 0
            ));
      }



      /*!
       * @brief Sum the elements of this vector
       * @return Sum of the values of the vectors scalar elements
       */
      RAJA_INLINE
      element_type sum(camp::idx_t N = 4) const
      {
        if(N <= 0){
          return element_type(0);
        }
        // swap pairs and add
        auto sh1 = permute<0x5>(m_value);
        auto red1 = _mm256_add_epi64(m_value, sh1);

        // add lower and upper
        return _mm256_extract_epi64(red1, 0) + _mm256_extract_epi64(red1, 2);
      }


      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_INLINE
      element_type max(camp::idx_t N = 4) const
      {
        if(N <= 0 || N > 4){
          return RAJA::operators::limits<long>::min();
        }

        // AVX2 does not supply an 64bit integer max?!?
        auto red = get(0);

        if(N > 1){
          auto v1 = get(1);
          red = red < v1 ? v1 : red;
        }
        if(N > 2){
          auto v2 = get(2);
          red = red < v2 ? v2 : red;
        }
        if(N > 3){
          auto v3 = get(3);
          red = red < v3 ? v3 : red;
        }

        return red;
      }

      /*!
       * @brief Returns element-wise largest values
       * @return Vector of the element-wise max values
       */
      RAJA_INLINE
      self_type vmax(self_type a) const
      {
          return self_type(_mm256_set_epi64x(
              get(3) > a.get(3) ? get(3) : a.get(3),
              get(2) > a.get(2) ? get(2) : a.get(2),
              get(1) > a.get(1) ? get(1) : a.get(1),
              get(0) > a.get(0) ? get(0) : a.get(0) ));
        
      }

      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_INLINE
      element_type min(camp::idx_t N = 4) const
      {
        if(N <= 0 || N > 4){
          return RAJA::operators::limits<long>::max();
        }

        // AVX2 does not supply an 64bit integer max?!?
        auto red = get(0);

        if(N > 1){
          auto v1 = get(1);
          red = red > v1 ? v1 : red;
        }
        if(N > 2){
          auto v2 = get(2);
          red = red > v2 ? v2 : red;
        }
        if(N > 3){
          auto v3 = get(3);
          red = red > v3 ? v3 : red;
        }

        return red;
      }

      /*!
       * @brief Returns element-wise largest values
       * @return Vector of the element-wise max values
       */
      RAJA_INLINE
      self_type vmin(self_type a) const
      {
          return self_type(_mm256_set_epi64x(
              get(3) < a.get(3) ? get(3) : a.get(3),
              get(2) < a.get(2) ? get(2) : a.get(2),
              get(1) < a.get(1) ? get(1) : a.get(1),
              get(0) < a.get(0) ? get(0) : a.get(0) ));
        
      }
  };



}  // namespace RAJA


#endif

#endif //__AVX2__
