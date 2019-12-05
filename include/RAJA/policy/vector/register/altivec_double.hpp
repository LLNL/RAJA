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

#ifdef __ALTIVEC__

#ifndef RAJA_policy_vector_register_altivec_double_HPP
#define RAJA_policy_vector_register_altivec_double_HPP

#include "RAJA/config.hpp"
#include "RAJA/util/macros.hpp"

// Include SIMD intrinsics header file
#include <altivec.h>
#include <cmath>


namespace RAJA
{


  template<>
  class Register<vector_altivec_register, double, 2>{
    public:
      using self_type = Register<vector_altivec_register, double, 2>;
      using element_type = double;

      static constexpr size_t s_num_elem = 2;
      static constexpr size_t s_byte_width = s_num_elem*sizeof(double);
      static constexpr size_t s_bit_width = s_byte_width*8;


    private:
      vector double m_value;


    public:

      using simd_type = decltype(m_value);

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
      explicit Register(simd_type const &c) : m_value(c) {}


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
       * @brief Load constructor, assuming scalars are in consecutive memory
       * locations.
       */
      RAJA_INLINE
      void load(element_type const *ptr){
        m_value = *((simd_type const *)ptr);
      }

      /*!
       * @brief Strided load constructor, when scalars are located in memory
       * locations ptr, ptr+stride, ptr+2*stride, etc.
       *
       *
       * Note: this could be done with "gather" instructions if they are
       * available. (like in avx2, but not in avx)
       */
      RAJA_INLINE
      void load(element_type const *ptr, size_t stride){
        if(stride == 1){
          load(ptr);
        }
        else{
          m_value = simd_type{ptr[0], ptr[stride]};
        }
      }



      /*!
       * @brief Store operation, assuming scalars are in consecutive memory
       * locations.
       */
      RAJA_INLINE
      void store(element_type *ptr) const{
        *((simd_type *)ptr) = m_value;
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
          store(ptr);
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
      element_type operator[](IDX i) const
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

      /*!
       * @brief Set entire vector to a single scalar value
       * @param value Value to set all vector elements to
       */
      RAJA_INLINE
      self_type const &operator=(element_type value)
      {
        m_value = simd_type(value);
        return *this;
      }

      /*!
       * @brief Assign one register to antoher
       * @param x Vector to copy
       * @return Value of (*this)
       */
      RAJA_INLINE
      self_type const &operator=(self_type const &x)
      {
        m_value = x.m_value;
        return *this;
      }


      /*!
       * @brief Add two vector registers
       * @param x Vector to add to this register
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      self_type operator+(self_type const &x) const
      {
        return self_type(m_value + x.m_value);
      }

      /*!
       * @brief Add a vector to this vector
       * @param x Vector to add to this register
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      self_type const &operator+=(self_type const &x)
      {
        m_value += x.m_value;
        return *this;
      }

      /*!
       * @brief Subtract two vector registers
       * @param x Vector to subctract from this register
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      self_type operator-(self_type const &x) const
      {
        return self_type(m_value - x.m_value);
      }

      /*!
       * @brief Subtract a vector from this vector
       * @param x Vector to subtract from this register
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      self_type const &operator-=(self_type const &x)
      {
        m_value -= x.m_value;
        return *this;
      }

      /*!
       * @brief Multiply two vector registers, element wise
       * @param x Vector to subctract from this register
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      self_type operator*(self_type const &x) const
      {
        return self_type(m_value * x.m_value);
      }

      /*!
       * @brief Multiply a vector with this vector
       * @param x Vector to multiple with this register
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      self_type const &operator*=(self_type const &x)
      {
        m_value *= x.m_value;
        return *this;
      }

      /*!
       * @brief Divide two vector registers, element wise
       * @param x Vector to subctract from this register
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      self_type operator/(self_type const &x) const
      {
        return self_type(m_value / x.m_value);
      }

      /*!
       * @brief Divide this vector by another vector
       * @param x Vector to divide by
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      self_type const &operator/=(self_type const &x)
      {
        m_value /= x.m_value;
        return *this;
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
        return self_type(simd_type{RAJA::max(m_value[0], a[0]), RAJA::max(m_value[1], a[1])});
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
        return self_type(simd_type{RAJA::min(m_value[0], a[0]), RAJA::min(m_value[1], a[1])});
      }
  };



}  // namespace RAJA


#endif

#endif //__AVX2__
