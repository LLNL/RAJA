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
#include "RAJA/pattern/vector.hpp"


// Include SIMD intrinsics header file
#include <altivec.h>
#include <cmath>



namespace RAJA
{

  template<>
  class Register<altivec_register, int>:
    public internal::RegisterBase<Register<altivec_register, int>>
  {
    public:
      using register_policy = altivec_register;
      using self_type = Register<altivec_register, int>;
      using element_type = int;



    private:
      vector int m_value;


    public:

      using register_type = decltype(m_value);

      static constexpr camp::idx_t s_num_elem = 4;

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
      explicit Register(register_type const &c) : m_value(c) {}


      /*!
       * @brief Copy constructor
       */
      RAJA_INLINE
      Register(self_type const &c) : m_value(c.m_value) {}


      /*!
       * @brief Construct from scalar.
       * Sets all elements to same value (broadcast).
       */
      RAJA_INLINE
      Register(element_type const &c) : m_value{c,c,c,c} {}


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
        if(N <= 0){
          m_value = register_type(0);
        }
        if(stride == 1){
          if(N == 4){
            m_value = *((register_type const *)ptr);
          }
          else{
            m_value = register_type{
              N >= 1 ? ptr[0] : 0,
              N >= 2 ? ptr[1] : 0,
              N >= 3 ? ptr[2] : 0,
              N >= 4 ? ptr[3] : 0};
          }
        }
        else{
          m_value = register_type{
                        N >= 1 ? ptr[0] : 0,
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
      self_type const &store(element_type *ptr, camp::idx_t stride = 1, camp::idx_t N = 4) const{
        if(N <= 0){
          return *this;
        }
        if(stride == 1 && N ==4){
          *((register_type *)ptr) = m_value;
        }
        else{
          for(camp::idx_t i = 0;i < N;++ i){
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
      self_type &broadcast(element_type const &c){
        m_value = register_type{c,c,c,c};
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
      self_type divide(self_type const &b, camp::idx_t N = 4) const {
        if(N <= 0){
          return self_type();
        }
        if(N==4){
          return self_type(vec_div(m_value, b.m_value));
        }
        else{
          return self_type(
              register_type{
                N >= 1 ? m_value[0]/b.m_value[0] : 0,
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
      element_type sum(camp::idx_t N = 4) const
      {
        if(N <= 0){
          return element_type(0);
        }

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
      element_type max(camp::idx_t N = 4) const
      {
        if(N <= 0){
          return RAJA::operators::limits<element_type>::min();
        }
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
      element_type min(camp::idx_t N = 4) const
      {
        if(N <= 0){
          return RAJA::operators::limits<element_type>::max();
        }
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
