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



#ifndef RAJA_policy_vector_register_altivec_double_HPP
#define RAJA_policy_vector_register_altivec_double_HPP

#include "RAJA/config.hpp"

#ifdef RAJA_ALTIVEC

#include "RAJA/util/macros.hpp"
#include "RAJA/pattern/register.hpp"


// Include SIMD intrinsics header file
#include <altivec.h>
#include <cmath>



namespace RAJA
{


  template<>
  class Register<vector_altivec_register, double, 2>:
    public internal::RegisterBase<Register<vector_altivec_register, double, 2>>
  {
    public:
      using self_type = Register<vector_altivec_register, double, 2>;
      using element_type = double;

      static constexpr size_t s_num_elem = 2;
      static constexpr size_t s_byte_width = s_num_elem*sizeof(double);
      static constexpr size_t s_bit_width = s_byte_width*8;


    private:
      vector double m_value;


    public:

      using register_type = decltype(m_value);

      /*!
       * @brief Default constructor, zeros register contents
       */
      RAJA_INLINE
      Register() : m_value{0.0, 0.0} {
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
      Register(element_type const &c) : m_value{c, c} {}


      /*!
       * @brief Strided load constructor, when scalars are located in memory
       * locations ptr, ptr+stride, ptr+2*stride, etc.
       *
       *
       * Note: this could be done with "gather" instructions if they are
       * available. (like in avx2, but not in avx)
       */
      RAJA_INLINE
      void load(element_type const *ptr, size_t stride = 1){
        if(stride == 1){
          m_value = *((register_type const *)ptr);
        }
        else{
          m_value = register_type{ptr[0], ptr[stride]};
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
      void store(element_type *ptr, size_t stride = 1) const{
        if(stride == 1){
          *((register_type *)ptr) = m_value;
        }
        else{
          for(size_t i = 0;i < s_num_elem;++ i){
            ptr[i*stride] = m_value[i];
          }
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
      self_type broadcast(element_type const &a){
        return self_type(a);
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
        return self_type(vec_add(a.m_value, b.m_value));
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      self_type subtract(self_type const &a, self_type const &b){
        return self_type(vec_sub(a.m_value, b.m_value));
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      self_type multiply(self_type const &a, self_type const &b){
        return self_type(vec_mul(a.m_value, b.m_value));
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      self_type divide(self_type const &a, self_type const &b){
        return self_type(vec_div(a.m_value, b.m_value));
      }

      /*!
       * @brief Sum the elements of this vector
       * @return Sum of the values of the vectors scalar elements
       */
      RAJA_INLINE
      element_type sum() const
      {
        return m_value[0] + m_value[1];
      }

      /*!
       * @brief Dot product of two vectors
       * @param x Other vector to dot with this vector
       * @return Value of (*this) dot x
       */
      RAJA_INLINE
      element_type dot(self_type const &x) const
      {
        return self_type((*this) * x).sum();
      }

      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_INLINE
      element_type max() const
      {
        // take the minimum of a lower and upper lane
        return RAJA::max<double>(m_value[0], m_value[1]);
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
        // take the minimum of a lower and upper lane
        return RAJA::min<double>(m_value[0], m_value[1]);
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
