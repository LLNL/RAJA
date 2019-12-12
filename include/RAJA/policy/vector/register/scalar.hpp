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

#ifndef RAJA_policy_vector_register_scalar_HPP
#define RAJA_policy_vector_register_scalar_HPP

#include<RAJA/pattern/register.hpp>

namespace RAJA
{

  struct vector_scalar_register {};

  /**
   * A specialization for a single element register.
   * We will implement this as a scalar value, and let the compiler use
   * whatever registers it deems appropriate.
   */
  template<typename T>
  class Register<vector_scalar_register, T, 1> :
    public internal::RegisterBase<Register<vector_scalar_register, T, 1>>
  {
    public:
      using self_type = Register<vector_scalar_register, T, 1>;
      using element_type = T;
      using register_type = T;

      static constexpr size_t s_num_elem = 1;
      static constexpr size_t s_byte_width = sizeof(T);
      static constexpr size_t s_bit_width = s_byte_width*8;

    private:
      T m_value;

    public:

      /*!
       * @brief Default constructor, zeros register contents
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      Register() : m_value(0) {
      }

      /*!
       * @brief Copy constructor from underlying simd register
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      Register(element_type const &c) : m_value(c) {}


      /*!
       * @brief Copy constructor
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      Register(self_type const &c) : m_value(c.m_value) {}


      /*!
       * @brief Strided load constructor, when scalars are located in memory
       * locations ptr, ptr+stride, ptr+2*stride, etc.
       *
       *
       * Note: this could be done with "gather" instructions if they are
       * available. (like in avx2, but not in avx)
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      void load(element_type const *ptr, size_t = 1){
        m_value = ptr[0];
      }

      /*!
       * @brief Strided store operation, where scalars are stored in memory
       * locations ptr, ptr+stride, ptr+2*stride, etc.
       *
       *
       * Note: this could be done with "scatter" instructions if they are
       * available.
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      void store(element_type *ptr, size_t = 1) const{
        ptr[0] = m_value;
      }


      /*!
       * @brief Get scalar value from vector register
       * @param i Offset of scalar to get
       * @return Returns scalar value at i
       */
      template<typename IDX>
      constexpr
      RAJA_INLINE
      RAJA_HOST_DEVICE
      element_type get(IDX) const
      {return m_value;}


      /*!
       * @brief Set scalar value in vector register
       * @param i Offset of scalar to set
       * @param value Value of scalar to set
       */
      template<typename IDX>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      void set(IDX , element_type value)
      {m_value = value;}



      RAJA_HOST_DEVICE
      RAJA_INLINE
      void broadcast(element_type const &a){
        m_value = a;
      }


      RAJA_HOST_DEVICE
      RAJA_INLINE
      void copy(self_type const &src){
        m_value = src.m_value;
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type add(self_type const &b) const {
        return self_type(m_value + b.m_value);
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type subtract(self_type const &b) const {
        return self_type(m_value - b.m_value);
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type multiply(self_type const &b) const {
        return self_type(m_value * b.m_value);
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type divide(self_type const &b) const {
        return self_type(m_value / b.m_value);
      }



      /*!
       * @brief Sum the elements of this vector
       * @return Sum of the values of the vectors scalar elements
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      element_type sum() const
      {
        return m_value;
      }


      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      element_type max() const
      {
        return m_value;
      }

      /*!
       * @brief Returns element-wise largest values
       * @return Vector of the element-wise max values
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type vmax(self_type a) const
      {
        return self_type(RAJA::max<element_type>(m_value, a.m_value));
      }

      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type min() const
      {
        return m_value;
      }

      /*!
       * @brief Returns element-wise largest values
       * @return Vector of the element-wise max values
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type vmin(self_type a) const
      {
        return self_type(RAJA::min<element_type>(m_value, a.m_value));
      }

  };

}  // namespace RAJA


#endif
