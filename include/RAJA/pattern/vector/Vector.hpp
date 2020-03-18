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

#ifndef RAJA_pattern_vector_vector_HPP
#define RAJA_pattern_vector_vector_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/pattern/vector/VectorProductRef.hpp"

#include <array>

namespace RAJA
{


/*!
 * \file
 * Vector operation functions in the namespace RAJA

 *
 */

  template<typename REGISTER_TYPE, size_t NUM_ELEM, bool FIXED_LENGTH>
  class Vector;

  template<template<typename, typename, size_t> class REGISTER_TYPE, typename REGISTER_POLICY, typename ELEMENT_TYPE, size_t NUM_REG_ELEM, size_t NUM_ELEM, bool FIXED_LENGTH>
  class Vector<REGISTER_TYPE<REGISTER_POLICY, ELEMENT_TYPE, NUM_REG_ELEM>, NUM_ELEM, FIXED_LENGTH>
  {
    public:
      using full_register_type =
          REGISTER_TYPE<REGISTER_POLICY, ELEMENT_TYPE, NUM_REG_ELEM>;
      static constexpr camp::idx_t s_num_register_elem = NUM_REG_ELEM;

      using self_type = Vector<full_register_type, NUM_ELEM, FIXED_LENGTH>;
      using vector_type = self_type;
      using element_type = ELEMENT_TYPE;


      static constexpr camp::idx_t s_is_fixed = FIXED_LENGTH;

      static constexpr camp::idx_t s_num_elem = NUM_ELEM;
      static constexpr camp::idx_t s_byte_width = sizeof(element_type);
      static constexpr camp::idx_t s_bit_width = s_byte_width*8;


      static_assert(s_num_elem % s_num_register_elem == 0 || s_is_fixed,
          "Vector must use a whole number of registers if it's variable length");


      static constexpr camp::idx_t s_num_full_registers = s_num_elem / s_num_register_elem;

      static constexpr camp::idx_t s_num_full_elem = s_num_full_registers*s_num_register_elem;

      static constexpr camp::idx_t s_num_partial_registers =
          s_num_full_elem == s_num_elem ? 0 : 1;

      static constexpr camp::idx_t s_num_partial_elem = s_num_elem - s_num_full_elem;

      using partial_register_type =
          REGISTER_TYPE<REGISTER_POLICY, ELEMENT_TYPE, s_num_partial_elem ? s_num_partial_elem : 1>;

    private:
      /*
       * Note (AJK):
       * We make sure that we don't have "zero length arrays" which seems to make only the MSVC
       * compile croak... and all other compilers seem happy with.
       * I would expect that if the number of full or partial register is zero, and we
       * never touch them, that they should get optimized out anyways.
       */
      full_register_type m_full_registers[s_num_full_registers > 0 ? s_num_full_registers : 1];
      partial_register_type m_partial_register[s_num_partial_registers > 0 ? s_num_partial_registers : 1];

      camp::idx_t m_length;
    public:


      /*!
       * @brief Default constructor, zeros register contents
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      Vector() : m_length(s_num_elem){}

      /*!
       * @brief Copy constructor
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      Vector(self_type const &c) :
        m_length(c.m_length)
      {
        for(camp::idx_t i = 0;i < s_num_full_registers;++ i){
          m_full_registers[i] = c.m_full_registers[i];
        }
        if(s_num_partial_registers){
          m_partial_register[0] = c.m_partial_register[0];
        }
      }

      /*!
       * @brief Scalar constructor (broadcast)
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      Vector(element_type const &c)
      {
        for(camp::idx_t i = 0;i < s_num_full_registers;++ i){
          m_full_registers[i].broadcast(c);
        }
        if(s_num_partial_registers){
          m_partial_register[0].broadcast(c);
        }
        m_length = s_num_elem;
      }


      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      constexpr
      bool is_root() {
        return full_register_type::is_root();
      }



      /*!
       * @brief Strided load constructor, when scalars are located in memory
       * locations ptr, ptr+stride, ptr+2*stride, etc.
       *
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      void load(element_type const *ptr, camp::idx_t stride = 1){
        m_length = s_num_elem;
        for(camp::idx_t i = 0;i < s_num_full_registers;++ i){
          m_full_registers[i].load(ptr + i*stride*s_num_register_elem, stride);
        }
        if(s_num_partial_registers){
          m_partial_register[0].load(ptr + stride*s_num_full_elem, stride);
        }
      }

      /*!
       * @brief Load constructor, assuming scalars are in consecutive memory
       * locations.
       *
       * For fixed length vectors, the length arguments is ignored, otherwise
       * only the specified number of values is read in.
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      void load_n(element_type const *ptr, camp::idx_t length, camp::idx_t stride = 1){
        m_length = length;
        if(s_is_fixed || length == s_num_elem){
          load(ptr, stride);
        }
        else{
          for(camp::idx_t i = 0;i < length;++ i){
            set(i, ptr[i*stride]);
          }
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
      RAJA_HOST_DEVICE
      RAJA_INLINE
      void store(element_type *ptr, camp::idx_t stride = 1) const{
        if(s_is_fixed || m_length == s_num_elem){
          for(camp::idx_t i = 0;i < s_num_full_registers;++ i){
            m_full_registers[i].store(ptr + i*stride*s_num_register_elem, stride);
          }
          if(s_num_partial_registers){
            m_partial_register[0].store(ptr + stride*s_num_full_elem, stride);
          }
        }
        else{
          for(camp::idx_t i = 0;i < m_length;++ i){
            ptr[i*stride] = (*this)[i];
          }
        }
      }


      /*!
       * @brief Get scalar value from vector
       * This will not be the most efficient due to the offset calculation.
       * @param i Offset of scalar to get
       * @return Returns scalar value at i
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type operator[](camp::idx_t i) const
      {
        // compute the register
        camp::idx_t r = i/s_num_register_elem;

        // compute the element in the register (equiv: i % s_num_register_elem)
        camp::idx_t e = i - (r*s_num_register_elem);

        if(!s_is_fixed || r < s_num_full_registers){
          return m_full_registers[r][e];
        }
        else{
          return m_partial_register[0][e];
        }
      }


      /*!
       * @brief Set scalar value in vector register
       * @param i Offset of scalar to set
       * @param value Value of scalar to set
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      void set(camp::idx_t i, element_type value)
      {
        // compute the register
        camp::idx_t r = i/s_num_register_elem;

        // compute the element in the register (equiv: i % s_num_register_elem)
        camp::idx_t e = i - (r*s_num_register_elem);

        if(!s_is_fixed || r < s_num_full_registers){
          m_full_registers[r].set(e, value);
        }
        else{
          m_partial_register[0].set(e, value);
        }
      }

      /*!
       * @brief Set entire vector to a single scalar value
       * @param value Value to set all vector elements to
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator=(element_type value)
      {
        for(camp::idx_t i = 0;i < s_num_full_registers;++ i){
          m_full_registers[i] = value;
        }
        if(s_num_partial_registers){
          m_partial_register[0] = value;
        }
        m_length = s_num_elem;
      }

      /*!
       * @brief Assign one register to antoher
       * @param x Vector to copy
       * @return Value of (*this)
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator=(self_type const &x)
      {
        for(camp::idx_t i = 0;i < s_num_full_registers;++ i){
          m_full_registers[i] = x.m_full_registers[i];
        }
        if(s_is_fixed && s_num_partial_registers){
          m_partial_register[0] = x.m_partial_register[0];
        }
        m_length = x.m_length;

        return *this;
      }


      /*!
       * @brief Add two vector registers
       * @param x Vector to add to this register
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type operator+(self_type const &x) const
      {
        self_type result = *this;
        result += x;
        return result;
      }

      /*!
       * @brief Add a vector to this vector
       * @param x Vector to add to this register
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator+=(self_type const &x)
      {
        for(camp::idx_t i = 0;i < s_num_full_registers;++ i){
          m_full_registers[i] += x.m_full_registers[i];
        }
        if(s_is_fixed && s_num_partial_registers){
          m_partial_register[0] += x.m_partial_register[0];
        }
        m_length = RAJA::min(m_length, x.m_length);

        return *this;
      }

      /*!
       * @brief Subtract two vector registers
       * @param x Vector to subctract from this register
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type operator-(self_type const &x) const
      {
        self_type result = *this;
        result -= x;
        return result;
      }

      /*!
       * @brief Subtract a vector from this vector
       * @param x Vector to subtract from this register
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator-=(self_type const &x)
      {
        for(camp::idx_t i = 0;i < s_num_full_registers;++ i){
          m_full_registers[i] -= x.m_full_registers[i];
        }
        if(s_is_fixed && s_num_partial_registers){
          m_partial_register[0] -= x.m_partial_register[0];
        }
        m_length = RAJA::min(m_length, x.m_length);

        return *this;
      }

      /*!
       * @brief Multiply two vector registers, element wise
       * @param x Vector to subctract from this register
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      VectorProductRef<self_type> operator*(self_type const &x) const
      {
        return VectorProductRef<self_type>(*this, x);
      }

      /*!
       * @brief Multiply a vector with this vector
       * @param x Vector to multiple with this register
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator*=(self_type const &x)
      {
        for(camp::idx_t i = 0;i < s_num_full_registers;++ i){
          m_full_registers[i] *= x.m_full_registers[i];
        }
        if(s_is_fixed && s_num_partial_registers){
          m_partial_register[0] *= x.m_partial_register[0];
        }
        m_length = RAJA::min(m_length, x.m_length);

        return *this;
      }

      /*!
       * @brief Divide two vector registers, element wise
       * @param x Vector to subctract from this register
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type operator/(self_type const &x) const
      {
        self_type result = *this;
        result /= x;
        return result;
      }

      /*!
       * @brief Divide this vector by another vector
       * @param x Vector to divide by
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator/=(self_type const &x)
      {
        for(camp::idx_t i = 0;i < s_num_full_registers;++ i){
          m_full_registers[i] /= x.m_full_registers[i];
        }
        if(s_is_fixed && s_num_partial_registers){
          m_partial_register[0] /= x.m_partial_register[0];
        }
        m_length = RAJA::min(m_length, x.m_length);

        return *this;
      }


      /**
        * @brief Fused multiply add: fma(b, c) = (*this)*b+c
        *
        * Derived types can override this to implement intrinsic FMA's
        *
        * @param b Second product operand
        * @param c Sum operand
        * @return Value of (*this)*b+c
        */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type fused_multiply_add(self_type const &b, self_type const &c) const
      {
        self_type result = *this;
        for(camp::idx_t i = 0;i < s_num_full_registers;++ i){
          result.m_full_registers[i] =
          m_full_registers[i].fused_multiply_add(b.m_full_registers[i], c.m_full_registers[i]);
        }
        if(s_is_fixed && s_num_partial_registers){
          result.m_partial_register[0] =
          m_partial_register[0].fused_multiply_add(b.m_partial_register[0], c.m_partial_register[0]);
        }
        return result;
      }


      /*!
       * @brief Sum the elements of this vector
       * @return Sum of the values of the vectors scalar elements
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type sum() const
      {
        element_type result = (element_type)0;
        if(m_length == s_num_elem){
          for(camp::idx_t i = 0;i < s_num_full_registers;++ i){
            result += m_full_registers[i].sum();
          }
          if(s_num_partial_registers){
            result += m_partial_register[0].sum();
          }
        }
        else{
          for(camp::idx_t i = 0;i < m_length;++ i){
            result += (*this)[i];
          }
        }
        return result;
      }

      /*!
       * @brief Dot product of two vectors
       * @param x Other vector to dot with this vector
       * @return Value of (*this) dot x
       *
       * NOTE: we could really do something more optimized here!
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type dot(self_type const &x) const
      {
        self_type z = (*this) * x;
        return z.sum();
      }


      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type max() const
      {
        if(s_is_fixed || m_length == s_num_elem){
          if(s_num_full_registers == 0){
            return m_partial_register[0].max();
          }

          element_type result = (element_type)m_full_registers[0].max();
          for(camp::idx_t i = 1;i < s_num_full_registers;++ i){
            auto new_val = m_full_registers[i].max();
            result = result > new_val ? result : new_val;
          }
          if(s_num_partial_registers){
            auto new_val = m_partial_register[0].max();
            result = result > new_val ? result : new_val;
          }
          return result;
        }
        else{
          element_type result = (*this)[0];
          for(camp::idx_t i = 1;i < m_length;++ i){
            auto new_val = (*this)[i];
            result = result > new_val ? result : new_val;
          }
          return result;
        }
      }

      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type min() const
      {
        if(s_is_fixed || m_length == s_num_elem){
          if(s_num_full_registers == 0){
            return m_partial_register[0].min();
          }

          element_type result = (element_type)m_full_registers[0].min();
          for(camp::idx_t i = 1;i < s_num_full_registers;++ i){
            auto new_val = m_full_registers[i].min();
            result = result < new_val ? result : new_val;
          }
          if(s_num_partial_registers){
            auto new_val = m_partial_register[0].min();
            result = result < new_val ? result : new_val;
          }
          return result;
        }
        else{
          element_type result = (*this)[0];
          for(camp::idx_t i = 1;i < m_length;++ i){
            auto new_val = (*this)[i];
            result = result < new_val ? result : new_val;
          }
          return result;
        }
      }

  };



  template<typename REGISTER_TYPE, size_t NUM_ELEM>
  using FixedVectorExt = Vector<REGISTER_TYPE, NUM_ELEM, true>;

  template<typename REGISTER_TYPE, size_t NUM_ELEM>
  using StreamVectorExt = Vector<REGISTER_TYPE, NUM_ELEM, false>;


  template<typename ST, typename REGISTER_TYPE, size_t NUM_ELEM, bool FIXED_LENGTH>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  Vector<REGISTER_TYPE, NUM_ELEM, FIXED_LENGTH>
  operator+(ST x, Vector<REGISTER_TYPE, NUM_ELEM, FIXED_LENGTH> const &y){
    return Vector<REGISTER_TYPE, NUM_ELEM, FIXED_LENGTH>(x) + y;
  }

  template<typename ST, typename REGISTER_TYPE, size_t NUM_ELEM, bool FIXED_LENGTH>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  Vector<REGISTER_TYPE, NUM_ELEM, FIXED_LENGTH>
  operator-(ST x, Vector<REGISTER_TYPE, NUM_ELEM, FIXED_LENGTH> const &y){
    return Vector<REGISTER_TYPE, NUM_ELEM, FIXED_LENGTH>(x) - y;
  }

  template<typename ST, typename REGISTER_TYPE, size_t NUM_ELEM, bool FIXED_LENGTH>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  Vector<REGISTER_TYPE, NUM_ELEM, FIXED_LENGTH>
  operator*(ST x, Vector<REGISTER_TYPE, NUM_ELEM, FIXED_LENGTH> const &y){
    return Vector<REGISTER_TYPE, NUM_ELEM, FIXED_LENGTH>(x) * y;
  }

  template<typename ST, typename REGISTER_TYPE, size_t NUM_ELEM, bool FIXED_LENGTH>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  Vector<REGISTER_TYPE, NUM_ELEM, FIXED_LENGTH>
  operator/(ST x, Vector<REGISTER_TYPE, NUM_ELEM, FIXED_LENGTH> const &y){
    return Vector<REGISTER_TYPE, NUM_ELEM, FIXED_LENGTH>(x) / y;
  }

}  // namespace RAJA


#endif
