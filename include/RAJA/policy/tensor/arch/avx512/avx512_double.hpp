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

#ifdef __AVX512F__

#ifndef RAJA_policy_vector_register_avx512_double_HPP
#define RAJA_policy_vector_register_avx512_double_HPP

#include "RAJA/config.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/pattern/tensor/TensorRegister.hpp"

// Include SIMD intrinsics header file
#include <immintrin.h>
#include <cmath>


namespace RAJA
{


  template<camp::idx_t SKEW>
  class TensorRegister<avx512_register, double, VectorLayout, camp::idx_seq<8>, SKEW> :
    public internal::TensorRegisterBase<TensorRegister<avx512_register, double, VectorLayout, camp::idx_seq<8>, SKEW>>
  {
    public:
      using register_policy = avx512_register;
      using self_type = TensorRegister<avx512_register, double, VectorLayout, camp::idx_seq<8>, SKEW>;
      using element_type = double;
      using register_type = __m512d;


    private:
      register_type m_value;

      RAJA_INLINE
      __mmask8 createMask(camp::idx_t N) const {
        // Generate a mask
				switch(N){
					case 0: return __mmask8(0x00);
					case 1: return __mmask8(0x01);
					case 2: return __mmask8(0x03);
					case 3: return __mmask8(0x07);
					case 4: return __mmask8(0x0F);
					case 5: return __mmask8(0x1F);
					case 6: return __mmask8(0x3F);
					case 7: return __mmask8(0x7F);
					case 8: return __mmask8(0xFF);
				}
				return __mmask8(0);
      }

      RAJA_INLINE
      __m512i createStridedOffsets(camp::idx_t stride) const {
        // Generate a strided offset list
				auto vstride = _mm512_set1_epi64(stride);
				auto vseq = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
				return _mm512_mullo_epi64(vstride, vseq);
      }

    public:

      static constexpr camp::idx_t s_num_elem = 8;

      /*!
       * @brief Default constructor, zeros register contents
       */
			// AVX512F
      RAJA_INLINE
      TensorRegister() : m_value(_mm512_setzero_pd()) {
      }

      /*!
       * @brief Copy constructor from underlying simd register
       */
      RAJA_INLINE
      constexpr
      explicit TensorRegister(register_type const &c) : m_value(c) {}


      /*!
       * @brief Copy constructor
       */
      RAJA_INLINE
      constexpr
      TensorRegister(self_type const &c) : m_value(c.m_value) {}

      /*!
       * @brief Copy assignment constructor
       */
      RAJA_INLINE
      self_type &operator=(self_type const &c){
        m_value = c.m_value;
        return *this;
      }

      /*!
       * @brief Construct from scalar.
       * Sets all elements to same value (broadcast).
       */
			// AVX512F
      RAJA_INLINE
      TensorRegister(element_type const &c) : m_value(_mm512_set1_pd(c)) {}


      /*!
       * @brief Load a full register from a stride-one memory location
       *
       */
      RAJA_INLINE
      self_type &load_packed(element_type const *ptr){
			  // AVX512F
        m_value = _mm512_loadu_pd(ptr);
        return *this;
      }

      /*!
       * @brief Partially load a register from a stride-one memory location given
       *        a run-time number of elements.
       *
       */
      RAJA_INLINE
      self_type &load_packed_n(element_type const *ptr, camp::idx_t N){
			  // AVX512F
        m_value = _mm512_mask_loadu_pd(_mm512_setzero_pd(), createMask(N), ptr);
        return *this;
      }

      /*!
       * @brief Gather a full register from a strided memory location
       *
       */
      RAJA_INLINE
      self_type &load_strided(element_type const *ptr, camp::idx_t stride){
			  // AVX512F
        m_value = _mm512_i64gather_pd(createStridedOffsets(stride),
				                              ptr,
                                      sizeof(element_type));
        return *this;
      }


      /*!
       * @brief Partially load a register from a stride-one memory location given
       *        a run-time number of elements.
       *
       */
      RAJA_INLINE
      self_type &load_strided_n(element_type const *ptr, camp::idx_t stride, camp::idx_t N){
				// AVX512F
        m_value = _mm512_mask_i64gather_pd(_mm512_setzero_pd(),
                                      createMask(N),
                                      createStridedOffsets(stride),
                                      ptr,
                                      sizeof(element_type));
        return *this;
      }


      /*!
       * @brief Store entire register to consecutive memory locations
       *
       */
      RAJA_INLINE
      self_type const &store_packed(element_type *ptr) const{
				// AVX512F
        _mm512_storeu_pd(ptr, m_value);
        return *this;
      }

      /*!
       * @brief Store entire register to consecutive memory locations
       *
       */
      RAJA_INLINE
      self_type const &store_packed_n(element_type *ptr, camp::idx_t N) const{
				// AVX512F
        _mm512_mask_storeu_pd(ptr, createMask(N), m_value);
        return *this;
      }

      /*!
       * @brief Store entire register to consecutive memory locations
       *
       */
      RAJA_INLINE
      self_type const &store_strided(element_type *ptr, camp::idx_t stride) const{
				// AVX512F
				_mm512_i64scatter_pd(ptr, 
				                     createStridedOffsets(stride),
														 m_value,
														 sizeof(element_type));
        return *this;
      }


      /*!
       * @brief Store partial register to consecutive memory locations
       *
       */
      RAJA_INLINE
      self_type const &store_strided_n(element_type *ptr, camp::idx_t stride, camp::idx_t N) const{
				// AVX512F
				_mm512_mask_i64scatter_pd(ptr, 
                           				createMask(N),
				                          createStridedOffsets(stride),
																	m_value,
														      sizeof(element_type));
        return *this;
      }

      /*!
       * @brief Get scalar value from vector register
       * @param i Offset of scalar to get
       * @return Returns scalar value at i
       */
      RAJA_INLINE
      element_type get(camp::idx_t i) const
      {return m_value[i];}


      /*!
       * @brief Set scalar value in vector register
       * @param i Offset of scalar to set
       * @param value Value of scalar to set
       */
      RAJA_INLINE
      self_type &set(camp::idx_t i, element_type value)
      {
        m_value[i] = value;
        return *this;
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &broadcast(element_type const &value){
        m_value =  _mm512_set1_pd(value);
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
        return self_type(_mm512_add_pd(m_value, b.m_value));
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type subtract(self_type const &b) const {
        return self_type(_mm512_sub_pd(m_value, b.m_value));
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type multiply(self_type const &b) const {
        return self_type(_mm512_mul_pd(m_value, b.m_value));
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type divide(self_type const &b, camp::idx_t N = 8) const {
        return self_type(_mm512_maskz_div_pd(createMask(N), m_value, b.m_value));
      }

// only use FMA's if the compiler has them turned on
#ifdef __FMA__
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type fused_multiply_add(self_type const &b, self_type const &c) const
      {
        return self_type(_mm512_fmadd_pd(m_value, b.m_value, c.m_value));
      }

      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type fused_multiply_subtract(self_type const &b, self_type const &c) const
      {
        return self_type(_mm512_fmsub_pd(m_value, b.m_value, c.m_value));
      }
#endif

      /*!
       * @brief Sum the elements of this vector
       * @return Sum of the values of the vectors scalar elements
       */
      RAJA_INLINE
      element_type sum(camp::idx_t N = 8) const
      {
				return _mm512_mask_reduce_add_pd(createMask(N), m_value);
      }


      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_INLINE
      element_type max(camp::idx_t N = 8) const
      {
				return _mm512_mask_reduce_max_pd(createMask(N), m_value);
      }

      /*!
       * @brief Returns element-wise largest values
       * @return Vector of the element-wise max values
       */
      RAJA_INLINE
      self_type vmax(self_type a) const
      {
        return self_type(_mm512_max_pd(m_value, a.m_value));
      }

      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_INLINE
      element_type min(camp::idx_t N = 8) const
      {
				return _mm512_mask_reduce_min_pd(createMask(N), m_value);
      }

      /*!
       * @brief Returns element-wise largest values
       * @return Vector of the element-wise max values
       */
      RAJA_INLINE
      self_type vmin(self_type a) const
      {
        return self_type(_mm512_min_pd(m_value, a.m_value));
      }
  };



}  // namespace RAJA


#endif

#endif //__AVX512F__
