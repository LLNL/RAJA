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



#ifndef RAJA_policy_vector_register_altivec_int32_HPP
#define RAJA_policy_vector_register_altivec_int32_HPP

#include "RAJA/config.hpp"
#ifdef RAJA_ALTIVEC

#include "RAJA/util/macros.hpp"
#include "RAJA/pattern/register.hpp"


// Include SIMD intrinsics header file
#include <altivec.h>
#include <cmath>



namespace RAJA
{


  template<size_t N>
  class Register<altivec_register, int, N>:
    public internal::RegisterBase<Register<altivec_register, int, N>>
  {

    static_assert(N >= 1, "Vector must have at least 1 lane");
    static_assert(N <= 4, "AltiVec can only have 4 lanes of 32-bit ints");

    public:
      using self_type = Register<altivec_register, int, N>;
      using element_type = int;

      static constexpr size_t s_num_elem = N;


    private:
      vector int m_value;


    public:

      using register_type = decltype(m_value);

      /*!
       * @brief Default constructor, zeros register contents
       */
      RAJA_INLINE
      Register() : m_value{0, 0, 0, 0} {
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
      Register(element_type const &c) :
        m_value{c,
                N >= 2 ? c : 0,
                N >= 3 ? c : 0,
                N >= 4 ? c : 0}
      {}


      /*!
       * @brief Strided load constructor, when scalars are located in memory
       * locations ptr, ptr+stride, ptr+2*stride, etc.
       *
       *
       * Note: this could be done with "gather" instructions if they are
       * available. (like in avx2, but not in avx)
       */
      RAJA_INLINE
      self_type &load(element_type const *ptr, size_t stride = 1){
        if(stride == 1){
          if(N == 4){
            m_value = *((register_type const *)ptr);
          }
          else{
            m_value = register_type{
              ptr[0],
              N >= 2 ? ptr[1] : 0,
              N >= 3 ? ptr[2] : 0,
              N >= 4 ? ptr[3] : 0};
          }
        }
        else{
          m_value = register_type{
                        ptr[0],
                        N >= 2 ? ptr[stride] : 0,
                        N >= 3 ? ptr[2*stride] : 0,
                        N >= 4 ? ptr[3*stride] : 0};
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
      self_type const &store(element_type *ptr, size_t stride = 1) const{
        if(stride == 1 && N ==4){
          *((register_type *)ptr) = m_value;
        }
        else{
          for(size_t i = 0;i < N;++ i){
            ptr[i*stride] = m_value[i];
          }
        }
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
      element_type get(IDX i) const
      {return m_value[i];}


      /*!
       * @brief Set scalar value in vector register
       * @param i Offset of scalar to set
       * @param value Value of scalar to set
       */
      template<typename IDX>
      RAJA_INLINE
      self_type &set(IDX i, element_type value)
      {
        m_value[i] = value;
        return *this;
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &broadcast(element_type const &c){
        m_value = register_type{c,
                N >= 2 ? c : 0,
                N >= 3 ? c : 0,
                N >= 4 ? c : 0};
        return * this;
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
        return self_type(vec_add(m_value, b.m_value));
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type subtract(self_type const &b) const {
        return self_type(vec_sub(m_value, b.m_value));
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type multiply(self_type const &b) const {
        return self_type(vec_mul(m_value, b.m_value));
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type divide(self_type const &b) const {
        if(N==4){
          return self_type(vec_div(m_value, b.m_value));
        }
        else{
          return self_type(
              register_type{
                m_value[0]/b.m_value[0],
                N >= 2 ? m_value[1]/b.m_value[1] : 0,
                N >= 3 ? m_value[2]/b.m_value[2] : 0,
                N >= 4 ? m_value[3]/b.m_value[3] : 0}
          );
        }
      }



      /*!
       * @brief Sum the elements of this vector
       * @return Sum of the values of the vectors scalar elements
       */
      RAJA_INLINE
      element_type sum() const
      {
        vector unsigned char perm1 {
          0x04, 0x05, 0x06, 0x07,  // m_value[1]
          0x00, 0x01, 0x02, 0x03,  // m_value[0]
          0x0C, 0x0D, 0x0E, 0x0F,  // m_value[3]
          0x08, 0x09, 0x0A, 0x0B}; // m_value[2]
        auto sh1 = vec_perm(m_value, m_value, perm1);

        auto red1 = vec_add(m_value, sh1);

        vector unsigned char perm2 {
          0x0C, 0x0D, 0x0E, 0x0F,  // m_value[3]
          0x08, 0x09, 0x0A, 0x0B,  // m_value[2]
          0x00, 0x01, 0x02, 0x03,  // m_value[0]
          0x04, 0x05, 0x06, 0x07};  // m_value[1]

        auto sh2 = vec_perm(red1, red1, perm2);

        auto red2 = vec_add(red1, sh2);

        return red2[0];
      }


      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_INLINE
      element_type max() const
      {
        if(N == 1){
          return m_value[0];
        }

        vector unsigned char perm1 {
          0x04, 0x05, 0x06, 0x07,  // m_value[1]
          0x00, 0x01, 0x02, 0x03,  // m_value[0]
          0x0C, 0x0D, 0x0E, 0x0F,  // m_value[3]
          0x08, 0x09, 0x0A, 0x0B}; // m_value[2]
        auto sh1 = vec_perm(m_value, m_value, perm1);

        auto red1 = vec_max(m_value, sh1);

        if(N == 2){
          return red1[0];
        }

        if(N == 3){
          return RAJA::max<element_type>(red1[0], m_value[2]);
        }
        vector unsigned char perm2 {
          0x08, 0x09, 0x0A, 0x0B,  // m_value[2]
          0x0C, 0x0D, 0x0E, 0x0F,  // m_value[3]
          0x00, 0x01, 0x02, 0x03,  // m_value[0]
          0x04, 0x05, 0x06, 0x07};  // m_value[1]

        auto sh2 = vec_perm(red1, red1, perm2);

        auto red2 = vec_max(red1, sh2);

        return red2[0];

      }

      /*!
       * @brief Returns element-wise largest values
       * @return Vector of the element-wise max values
       */
      RAJA_INLINE
      self_type vmax(self_type a) const
      {
        return self_type(vec_max(m_value, a.m_value));
      }

      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_INLINE
      element_type min() const
      {
        if(N == 1){
          return m_value[0];
        }

        vector unsigned char perm1 {
          0x04, 0x05, 0x06, 0x07,  // m_value[1]
          0x00, 0x01, 0x02, 0x03,  // m_value[0]
          0x0C, 0x0D, 0x0E, 0x0F,  // m_value[3]
          0x08, 0x09, 0x0A, 0x0B}; // m_value[2]
        auto sh1 = vec_perm(m_value, m_value, perm1);

        auto red1 = vec_min(m_value, sh1);

        if(N == 2){
          return red1[0];
        }

        if(N == 3){
          return RAJA::min<element_type>(red1[0], m_value[2]);
        }
        vector unsigned char perm2 {
          0x08, 0x09, 0x0A, 0x0B,  // m_value[2]
          0x0C, 0x0D, 0x0E, 0x0F,  // m_value[3]
          0x00, 0x01, 0x02, 0x03,  // m_value[0]
          0x04, 0x05, 0x06, 0x07};  // m_value[1]

        auto sh2 = vec_perm(red1, red1, perm2);

        auto red2 = vec_min(red1, sh2);

        return red2[0];
      }

      /*!
       * @brief Returns element-wise largest values
       * @return Vector of the element-wise max values
       */
      RAJA_INLINE
      self_type vmin(self_type a) const
      {
        return self_type(vec_min(m_value, a.m_value));
      }
  };



}  // namespace RAJA


#endif // RAJA_ALTIVEC

#endif // guard
