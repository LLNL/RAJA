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
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifdef __AVX2__

#ifndef RAJA_policy_vector_register_avx2_double_HPP
#define RAJA_policy_vector_register_avx2_double_HPP

#include "RAJA/config.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/pattern/tensor/internal/RegisterBase.hpp"

// Include SIMD intrinsics header file
#include <immintrin.h>
#include <cmath>


namespace RAJA
{
namespace expt
{

  template<>
  class Register<double, avx2_register> :
    public internal::expt::RegisterBase<Register<double, avx2_register>>
  {
    public:
      using base_type = internal::expt::RegisterBase<Register<double, avx2_register>>;

      using register_policy = avx2_register;
      using self_type = Register<double, avx2_register>;
      using element_type = double;
      using register_type = __m256d;

      using int_vector_type = Register<int64_t, avx2_register>;

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

    public:

      static constexpr camp::idx_t s_num_elem = 4;

      /*!
       * @brief Default constructor, zeros register contents
       */
      RAJA_INLINE
      Register() : m_value(_mm256_setzero_pd()) {
      }

      /*!
       * @brief Construct register with explicit values
       */
      RAJA_INLINE
      Register(element_type x0,
                     element_type x1,
                     element_type x2,
                     element_type x3) :
        m_value(_mm256_set_pd(x3,x2,x1,x0))
      {}


      /*!
       * @brief Copy constructor from underlying simd register
       */
      RAJA_INLINE
      explicit Register(register_type const &c) : m_value(c) {}


      /*!
       * @brief Copy constructor
       */
      RAJA_INLINE
      Register(self_type const &c) : base_type(c), m_value(c.m_value) {}

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
      RAJA_INLINE
      Register(element_type const &c) : m_value(_mm256_set1_pd(c)) {}


      /*!
       * @brief Returns underlying SIMD register.
       */
      RAJA_INLINE
      constexpr
      register_type get_register() const {
        return m_value;
      }



      /*!
       * @brief Load a full register from a stride-one memory location
       *
       */
      RAJA_INLINE
      self_type &load_packed(element_type const *ptr){
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_load_packed ++;
#endif
        m_value = _mm256_loadu_pd(ptr);
        return *this;
      }

      /*!
       * @brief Partially load a register from a stride-one memory location given
       *        a run-time number of elements.
       *
       */
      RAJA_INLINE
      self_type &load_packed_n(element_type const *ptr, camp::idx_t N){
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_load_packed_n ++;
#endif
        m_value = _mm256_maskload_pd(ptr, createMask(N));
        return *this;
      }

      /*!
       * @brief Gather a full register from a strided memory location
       *
       */
      RAJA_INLINE
      self_type &load_strided(element_type const *ptr, camp::idx_t stride){
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_load_strided ++;
#endif
        m_value = _mm256_i64gather_pd(ptr,
                                      createStridedOffsets(stride),
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
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_load_strided_n ++;
#endif
        m_value = _mm256_mask_i64gather_pd(_mm256_setzero_pd(),
                                      ptr,
                                      createStridedOffsets(stride),
                                      _mm256_castsi256_pd(createMask(N)),
                                      sizeof(element_type));
        return *this;
      }

      /*!
       * @brief Generic gather operation for full vector.
       *
       * Must provide another register containing offsets of all values
       * to be loaded relative to supplied pointer.
       *
       * Offsets are element-wise, not byte-wise.
       *
       */
      RAJA_INLINE
      self_type &gather(element_type const *ptr, int_vector_type offsets){
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_load_strided_n ++;
#endif
        m_value = _mm256_i64gather_pd(ptr,
                                      offsets.get_register(),
                                      sizeof(element_type));
        return *this;
      }

      /*!
       * @brief Generic gather operation for n-length subvector.
       *
       * Must provide another register containing offsets of all values
       * to be loaded relative to supplied pointer.
       *
       * Offsets are element-wise, not byte-wise.
       *
       */
      RAJA_INLINE
      self_type &gather_n(element_type const *ptr, int_vector_type offsets, camp::idx_t N){
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_load_strided_n ++;
#endif
        m_value = _mm256_mask_i64gather_pd(_mm256_setzero_pd(),
                                      ptr,
                                      offsets.get_register(),
                                      _mm256_castsi256_pd(createMask(N)),
                                      sizeof(element_type));
        return *this;
      }


      /*!
       * @brief Store entire register to consecutive memory locations
       *
       */
      RAJA_INLINE
      self_type const &store_packed(element_type *ptr) const{
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_store_packed ++;
#endif
        _mm256_storeu_pd(ptr, m_value);
        return *this;
      }

      /*!
       * @brief Store entire register to consecutive memory locations
       *
       */
      RAJA_INLINE
      self_type const &store_packed_n(element_type *ptr, camp::idx_t N) const{
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_store_packed_n ++;
#endif
        _mm256_maskstore_pd(ptr, createMask(N), m_value);
        return *this;
      }

      /*!
       * @brief Store entire register to consecutive memory locations
       *
       */
      RAJA_INLINE
      self_type const &store_strided(element_type *ptr, camp::idx_t stride) const{
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_store_strided ++;
#endif
        for(camp::idx_t i = 0;i < 4;++ i){
          ptr[i*stride] = m_value[i];
        }
        return *this;
      }


      /*!
       * @brief Store partial register to consecutive memory locations
       *
       */
      RAJA_INLINE
      self_type const &store_strided_n(element_type *ptr, camp::idx_t stride, camp::idx_t N) const{
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_store_strided_n ++;
#endif
        for(camp::idx_t i = 0;i < N;++ i){
          ptr[i*stride] = m_value[i];
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
      {return m_value[i];}


      /*!
       * @brief Set scalar value in vector register
       * @param i Offset of scalar to set
       * @param value Value of scalar to set
       */
      RAJA_INLINE
      self_type &set(element_type value, camp::idx_t i)
      {
        m_value[i] = value;
        return *this;
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &broadcast(element_type const &value){
        m_value =  _mm256_set1_pd(value);
        return *this;
      }

      /*!
       * @brief Extracts a scalar value and broadcasts to a new register
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type get_and_broadcast(int i) const {
        switch(i){
          case 0: return self_type(_mm256_permute4x64_pd (m_value, 0x00));
          case 1: return self_type(_mm256_permute4x64_pd (m_value, 0x55));
          case 2: return self_type(_mm256_permute4x64_pd (m_value, 0xAA));
          case 3: return self_type(_mm256_permute4x64_pd (m_value, 0xFF));
        }
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
        return self_type(_mm256_add_pd(m_value, b.m_value));
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type subtract(self_type const &b) const {
        return self_type(_mm256_sub_pd(m_value, b.m_value));
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type multiply(self_type const &b) const {
        return self_type(_mm256_mul_pd(m_value, b.m_value));
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type divide(self_type const &b) const {
        return self_type(_mm256_div_pd(m_value, b.m_value));
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type divide_n(self_type const &b, camp::idx_t N) const {
        // AVX2 does not supply a masked divide, so do it manually
        return self_type(_mm256_set_pd(
            N >= 4 ? get(3)/b.get(3) : 0,
            N >= 3 ? get(2)/b.get(2) : 0,
            N >= 2 ? get(1)/b.get(1) : 0,
            N >= 1 ? get(0)/b.get(0) : 0
            ));
      }

// only use FMA's if the compiler has them turned on
#ifdef __FMA__
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type multiply_add(self_type const &b, self_type const &c) const
      {
        return self_type(_mm256_fmadd_pd(m_value, b.m_value, c.m_value));
      }

      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type multiply_subtract(self_type const &b, self_type const &c) const
      {
        return self_type(_mm256_fmsub_pd(m_value, b.m_value, c.m_value));
      }
#endif

      /*!
       * @brief Sum the elements of this vector
       * @return Sum of the values of the vectors scalar elements
       */
      RAJA_INLINE
      element_type sum(camp::idx_t = 4) const
      {
        auto sh1 = _mm256_permute_pd(m_value, 0x5);
        auto red1 = _mm256_add_pd(m_value, sh1);
        return red1[0]+red1[2];
      }


      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_INLINE
      element_type max(camp::idx_t N = 4) const
      {
        if(N == 4){
          // permute the first two and last two lanes of the register
          // A = { v[1], v[0], v[3], v[2] }
          register_type a = _mm256_shuffle_pd(m_value, m_value, 0x5);

          // take the maximum value of each lane
          // B = { max{v[0], v[1]},
          //       max{v[0], v[1]},
          //       max{v[2], v[3]},
          //       max{v[2], v[3]} }
          register_type b = _mm256_max_pd(m_value, a);

          // now take the maximum of a lower and upper halves
          return RAJA::max<element_type>(b[0], b[2]);
        }
        else if(N == 3){
          // permute the first two and last two lanes of the register
          // use the third element TWICE, so we effectively remove the 4th
          // lane
          // A = { v[1], v[0], v[2], v[2] }
          register_type a = _mm256_shuffle_pd(m_value, m_value, 0x3);

          // take the maximum value of each lane
          // B = { max{v[0], v[1]},
          //       max{v[0], v[1]},
          //       max{v[2], v[2]},   <-- just v[2]
          //       max{v[2], v[3]} }
          register_type b = _mm256_max_pd(m_value, a);

          // now take the maximum of a lower and upper lane
          return RAJA::max<element_type>(b[0], b[2]);
        }
        else if(N == 2){
          return RAJA::max<element_type>(m_value[0], m_value[1]);
        }
        else if(N == 1){
          return m_value[0];
        }
        return RAJA::operators::limits<double>::min();
      }

      /*!
       * @brief Returns element-wise largest values
       * @return Vector of the element-wise max values
       */
      RAJA_INLINE
      self_type vmax(self_type a) const
      {
        return self_type(_mm256_max_pd(m_value, a.m_value));
      }

      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_INLINE
      element_type min() const
      {
        // permute the first two and last two lanes of the register
        // A = { v[1], v[0], v[3], v[2] }
        register_type a = _mm256_shuffle_pd(m_value, m_value, 0x5);

        // take the minimum value of each lane
        // B = { min{v[0], v[1]},
        //       min{v[0], v[1]},
        //       min{v[2], v[3]},
        //       min{v[2], v[3]} }
        register_type b = _mm256_min_pd(m_value, a);

        // now take the minimum of a lower and upper halves
        return RAJA::min<element_type>(b[0], b[2]);
      }

      /*!
       * @brief Returns the largest element from first N lanes
       * @return The largest scalar element in the register
       */
      RAJA_INLINE
      element_type min_n(camp::idx_t N) const
      {
        if(N == 4){
          // permute the first two and last two lanes of the register
          // A = { v[1], v[0], v[3], v[2] }
          register_type a = _mm256_shuffle_pd(m_value, m_value, 0x5);

          // take the minimum value of each lane
          // B = { min{v[0], v[1]},
          //       min{v[0], v[1]},
          //       min{v[2], v[3]},
          //       min{v[2], v[3]} }
          register_type b = _mm256_min_pd(m_value, a);

          // now take the minimum of a lower and upper halves
          return std::min<element_type>(b[0], b[2]);
        }
        else if(N == 3){
          // permute the first two and last two lanes of the register
          // use the third element TWICE, so we effectively remove the 4th
          // lane
          // A = { v[1], v[0], v[2], v[2] }
          register_type a = _mm256_shuffle_pd(m_value, m_value, 0x3);

          // take the minimum value of each lane
          // B = { min{v[0], v[1]},
          //       min{v[0], v[1]},
          //       min{v[2], v[2]},   <-- just v[2]
          //       min{v[2], v[3]} }
          register_type b = _mm256_min_pd(m_value, a);

          // now take the minimum of a lower and upper lane
          return std::min<element_type>(b[0], b[2]);
        }
        else if(N == 2){
          return std::min<element_type>(m_value[0], m_value[1]);
        }
        else if(N == 1){
          return m_value[0];
        }
        return RAJA::operators::limits<double>::max();
      }

      /*!
       * @brief Returns element-wise largest values
       * @return Vector of the element-wise max values
       */
      RAJA_INLINE
      self_type vmin(self_type a) const
      {
        return self_type(_mm256_min_pd(m_value, a.m_value));
      }
  };


}   // namespace expt

}  // namespace RAJA


#endif

#endif //__AVX2__
