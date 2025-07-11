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
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifdef __AVX2__

#ifndef RAJA_policy_vector_register_avx2_int64_HPP
#define RAJA_policy_vector_register_avx2_int64_HPP

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
class Register<int64_t, avx2_register>
    : public internal::expt::RegisterBase<Register<int64_t, avx2_register>>
{
public:
  using base_type =
      internal::expt::RegisterBase<Register<int64_t, avx2_register>>;

  using register_policy = avx2_register;
  using self_type       = Register<int64_t, avx2_register>;
  using element_type    = int64_t;
  using register_type   = __m256i;

  using int_vector_type = Register<int64_t, avx2_register>;

private:
  register_type m_value;

  RAJA_INLINE
  __m256i createMask(camp::idx_t N) const
  {
    // Generate a mask
    return _mm256_set_epi64x(N >= 4 ? -1 : 0, N >= 3 ? -1 : 0, N >= 2 ? -1 : 0,
                             N >= 1 ? -1 : 0);
  }

  RAJA_INLINE
  __m256i createStridedOffsets(camp::idx_t stride) const
  {
    // Generate a strided offset list
    return _mm256_set_epi64x(3 * stride, 2 * stride, stride, 0);
  }

  /*
   * Use the packed-double permute function because there isn't one
   * specifically for int64
   *
   * Just adds a bunch of casting, should be same cost
   */
  template<int perm>
  RAJA_INLINE __m256i permute(__m256i x) const
  {
    return _mm256_castpd_si256(_mm256_permute_pd(_mm256_castsi256_pd(x), perm));
  }

public:
  static constexpr camp::idx_t s_num_elem = 4;

  /*!
   * @brief Default constructor, zeros register contents
   */
  RAJA_INLINE
  Register() : m_value(_mm256_setzero_si256()) {}

  /*!
   * @brief Construct register with explicit values
   */
  RAJA_INLINE
  Register(element_type x0, element_type x1, element_type x2, element_type x3)
      : m_value(_mm256_set_epi64x(x3, x2, x1, x0))
  {}

  /*!
   * @brief Copy constructor from underlying simd register
   */
  RAJA_INLINE
  explicit Register(register_type const& c) : m_value(c) {}

  /*!
   * @brief Copy constructor
   */
  RAJA_INLINE
  Register(self_type const& c) : base_type(c), m_value(c.m_value) {}

  /*!
   * @brief Copy assignment constructor
   */
  RAJA_INLINE
  self_type& operator=(self_type const& c)
  {
    m_value = c.m_value;
    return *this;
  }

  /*!
   * @brief Construct from scalar.
   * Sets all elements to same value (broadcast).
   */
  RAJA_INLINE
  Register(element_type const& c) : m_value(_mm256_set1_epi64x(c)) {}

  /*!
   * @brief Returns underlying SIMD register.
   */
  RAJA_INLINE
  constexpr register_type get_register() const { return m_value; }

  /*!
   * @brief Load a full register from a stride-one memory location
   *
   */
  RAJA_INLINE
  self_type& load_packed(element_type const* ptr)
  {
    m_value = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(ptr));
    return *this;
  }

  /*!
   * @brief Partially load a register from a stride-one memory location given
   *        a run-time number of elements.
   *
   */
  RAJA_INLINE
  self_type& load_packed_n(element_type const* ptr, camp::idx_t N)
  {
    m_value = _mm256_castpd_si256(_mm256_maskload_pd(
        reinterpret_cast<double const*>(ptr), createMask(N)));
    return *this;
  }

  /*!
   * @brief Gather a full register from a strided memory location
   *
   */
  RAJA_INLINE
  self_type& load_strided(int64_t const* ptr, camp::idx_t stride)
  {
    m_value = _mm256_i64gather_epi64(reinterpret_cast<long long const*>(ptr),
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
  self_type& load_strided_n(element_type const* ptr,
                            camp::idx_t stride,
                            camp::idx_t N)
  {
    m_value = _mm256_mask_i64gather_epi64(
        _mm256_set1_epi64x(0), reinterpret_cast<long long const*>(ptr),
        createStridedOffsets(stride), createMask(N), sizeof(element_type));
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
  self_type& gather(element_type const* ptr, int_vector_type offsets)
  {
#ifdef RAJA_ENABLE_VECTOR_STATS
    RAJA::tensor_stats::num_vector_load_strided_n++;
#endif
    m_value =
        _mm256_i64gather_epi64(reinterpret_cast<long long const*>(ptr),
                               offsets.get_register(), sizeof(element_type));
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
  self_type& gather_n(element_type const* ptr,
                      int_vector_type offsets,
                      camp::idx_t N)
  {
#ifdef RAJA_ENABLE_VECTOR_STATS
    RAJA::tensor_stats::num_vector_load_strided_n++;
#endif
    m_value = _mm256_mask_i64gather_epi64(
        _mm256_setzero_si256(), reinterpret_cast<long long const*>(ptr),
        offsets.get_register(), createMask(N), sizeof(element_type));
    return *this;
  }

  /*!
   * @brief Store entire register to consecutive memory locations
   *
   */
  RAJA_INLINE
  self_type const& store_packed(element_type* ptr) const
  {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), m_value);
    return *this;
  }

  /*!
   * @brief Store entire register to consecutive memory locations
   *
   */
  RAJA_INLINE
  self_type const& store_packed_n(element_type* ptr, camp::idx_t N) const
  {
    _mm256_maskstore_epi64(reinterpret_cast<long long*>(ptr), createMask(N),
                           m_value);
    return *this;
  }

  /*!
   * @brief Store entire register to consecutive memory locations
   *
   */
  RAJA_INLINE
  self_type const& store_strided(element_type* ptr, camp::idx_t stride) const
  {
    for (camp::idx_t i = 0; i < 4; ++i)
    {
      ptr[i * stride] = m_value[i];
    }
    return *this;
  }

  /*!
   * @brief Store partial register to consecutive memory locations
   *
   */
  RAJA_INLINE
  self_type const& store_strided_n(element_type* ptr,
                                   camp::idx_t stride,
                                   camp::idx_t N) const
  {
    for (camp::idx_t i = 0; i < N; ++i)
    {
      ptr[i * stride] = m_value[i];
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
    switch (i)
    {
      case 0:
        return _mm256_extract_epi64(m_value, 0);
      case 1:
        return _mm256_extract_epi64(m_value, 1);
      case 2:
        return _mm256_extract_epi64(m_value, 2);
      case 3:
        return _mm256_extract_epi64(m_value, 3);
    }
    return 0;
  }

  /*!
   * @brief Set scalar value in vector register
   * @param i Offset of scalar to set
   * @param value Value of scalar to set
   */
  RAJA_INLINE
  self_type& set(element_type value, camp::idx_t i)
  {
    // got to be a nicer way to do this!?!?
    switch (i)
    {
      case 0:
        m_value = _mm256_insert_epi64(m_value, value, 0);
        break;
      case 1:
        m_value = _mm256_insert_epi64(m_value, value, 1);
        break;
      case 2:
        m_value = _mm256_insert_epi64(m_value, value, 2);
        break;
      case 3:
        m_value = _mm256_insert_epi64(m_value, value, 3);
        break;
    }

    return *this;
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& broadcast(element_type const& value)
  {
    m_value = _mm256_set1_epi64x(value);
    return *this;
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& copy(self_type const& src)
  {
    m_value = src.m_value;
    return *this;
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type add(self_type const& b) const
  {
    return self_type(_mm256_add_epi64(m_value, b.m_value));
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type subtract(self_type const& b) const
  {
    return self_type(_mm256_sub_epi64(m_value, b.m_value));
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type multiply(self_type const& b) const
  {
    // AVX2 does not supply an int64_t multiply, so do it manually
    return self_type(_mm256_set_epi64x(get(3) * b.get(3), get(2) * b.get(2),
                                       get(1) * b.get(1), get(0) * b.get(0)));
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type divide(self_type const& b) const
  {
    // AVX2 does not supply an integer divide, so do it manually
    return self_type(_mm256_set_epi64x(get(3) / b.get(3), get(2) / b.get(2),
                                       get(1) / b.get(1), get(0) / b.get(0)));
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type divide_n(self_type const& b, camp::idx_t N) const
  {
    // AVX2 does not supply an integer divide, so do it manually
    return self_type(_mm256_set_epi64x(
        N >= 4 ? get(3) / b.get(3) : 0, N >= 3 ? get(2) / b.get(2) : 0,
        N >= 2 ? get(1) / b.get(1) : 0, N >= 1 ? get(0) / b.get(0) : 0));
  }

  /*!
   * @brief Sum the elements of this vector
   * @return Sum of the values of the vectors scalar elements
   */
  RAJA_INLINE
  element_type sum() const
  {

    // swap pairs and add
    auto sh1  = permute<0x5>(m_value);
    auto red1 = _mm256_add_epi64(m_value, sh1);

    // add lower and upper
    return _mm256_extract_epi64(red1, 0) + _mm256_extract_epi64(red1, 2);
  }

  /*!
   * @brief Returns the largest element
   * @return The largest scalar element in the register
   */
  RAJA_INLINE
  element_type max() const
  {
    // AVX2 does not supply an 64bit integer max?!?
    auto red = get(0);

    auto v1 = get(1);
    red     = red < v1 ? v1 : red;

    auto v2 = get(2);
    red     = red < v2 ? v2 : red;

    auto v3 = get(3);
    red     = red < v3 ? v3 : red;

    return red;
  }

  /*!
   * @brief Returns the largest element
   * @return The largest scalar element in the register
   */
  RAJA_INLINE
  element_type max_n(camp::idx_t N) const
  {
    if (N <= 0 || N > 4)
    {
      return RAJA::operators::limits<int64_t>::min();
    }

    // AVX2 does not supply an 64bit integer max?!?
    auto red = get(0);

    if (N > 1)
    {
      auto v1 = get(1);
      red     = red < v1 ? v1 : red;
    }
    if (N > 2)
    {
      auto v2 = get(2);
      red     = red < v2 ? v2 : red;
    }
    if (N > 3)
    {
      auto v3 = get(3);
      red     = red < v3 ? v3 : red;
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
    return self_type(_mm256_set_epi64x(get(3) > a.get(3) ? get(3) : a.get(3),
                                       get(2) > a.get(2) ? get(2) : a.get(2),
                                       get(1) > a.get(1) ? get(1) : a.get(1),
                                       get(0) > a.get(0) ? get(0) : a.get(0)));
  }

  /*!
   * @brief Returns the largest element
   * @return The largest scalar element in the register
   */
  RAJA_INLINE
  element_type min() const
  {
    // AVX2 does not supply an 64bit integer max?!?
    auto red = get(0);

    auto v1 = get(1);
    red     = red > v1 ? v1 : red;

    auto v2 = get(2);
    red     = red > v2 ? v2 : red;

    auto v3 = get(3);
    red     = red > v3 ? v3 : red;

    return red;
  }

  /*!
   * @brief Returns the largest element
   * @return The largest scalar element in the register
   */
  RAJA_INLINE
  element_type min_n(camp::idx_t N) const
  {
    if (N <= 0 || N > 4)
    {
      return RAJA::operators::limits<int64_t>::max();
    }

    // AVX2 does not supply an 64bit integer max?!?
    auto red = get(0);

    if (N > 1)
    {
      auto v1 = get(1);
      red     = red > v1 ? v1 : red;
    }
    if (N > 2)
    {
      auto v2 = get(2);
      red     = red > v2 ? v2 : red;
    }
    if (N > 3)
    {
      auto v3 = get(3);
      red     = red > v3 ? v3 : red;
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
    return self_type(_mm256_set_epi64x(get(3) < a.get(3) ? get(3) : a.get(3),
                                       get(2) < a.get(2) ? get(2) : a.get(2),
                                       get(1) < a.get(1) ? get(1) : a.get(1),
                                       get(0) < a.get(0) ? get(0) : a.get(0)));
  }
};


}  // namespace expt

}  // namespace RAJA


#endif

#endif  //__AVX2__
