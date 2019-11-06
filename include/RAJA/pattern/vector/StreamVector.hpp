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

#ifndef RAJA_pattern_vector_streamvector_HPP
#define RAJA_pattern_vector_streamvector_HPP

#include "RAJA/config.hpp"
#include "RAJA/util/macros.hpp"

namespace RAJA
{


/*!
 * \file
 * Vector operation functions in the namespace RAJA

 *
 */

  template<typename REGISTER_TYPE, size_t MAX_ELEM>
  class StreamVector;

  template<template<typename, typename, size_t> class REGISTER_TYPE, typename REGISTER_POLICY, typename ELEMENT_TYPE, size_t NUM_REG_ELEM, size_t MAX_ELEM>
  class StreamVector<REGISTER_TYPE<REGISTER_POLICY, ELEMENT_TYPE, NUM_REG_ELEM>, MAX_ELEM>
  {
    public:
      using register_type =
        REGISTER_TYPE<REGISTER_POLICY, ELEMENT_TYPE, NUM_REG_ELEM>;
      static constexpr size_t s_num_register_elem = NUM_REG_ELEM;

      using self_type = StreamVector<register_type, MAX_ELEM>;
      using element_type = ELEMENT_TYPE;

      static constexpr size_t s_num_elem = MAX_ELEM;
      static constexpr size_t s_num_registers =
          s_num_elem / s_num_register_elem;

      static_assert(s_num_elem % s_num_register_elem == 0,
          "StreamVector must use a whole number of registers");


    private:
      std::array<register_type, s_num_registers> m_registers;
      size_t m_length;

    public:


      /*!
       * @brief Default constructor, zeros register contents
       */
      RAJA_INLINE
      StreamVector() : m_length(s_num_elem) {}

      /*!
       * @brief Copy constructor
       */
      RAJA_INLINE
      StreamVector(self_type const &c) :
        m_registers(c.m_registers),
        m_length(c.m_length)
          {}


      /*!
       * @brief Get scalar value from vector
       * This will not be the most efficient due to the offset calculation.
       * @param i Offset of scalar to get
       * @return Returns scalar value at i
       */
      RAJA_INLINE
      element_type operator[](size_t i) const
      {
        // compute the register
        size_t r = i/s_num_register_elem;

        // compute the element in the register (equiv: i % s_num_register_elem)
        size_t e = i - (r*s_num_register_elem);

        return m_registers[r][e];
      }


      /*!
       * @brief Set scalar value in vector register
       * @param i Offset of scalar to set
       * @param value Value of scalar to set
       */
      RAJA_INLINE
      void set(size_t i, element_type value)
      {
        // compute the register
        size_t r = i/s_num_register_elem;

        // compute the element in the register (equiv: i % s_num_register_elem)
        size_t e = i - (r*s_num_register_elem);

        m_registers[r].set(e, value);
      }


      /*!
       * @brief Load constructor, assuming scalars are in consecutive memory
       * locations.
       */
      RAJA_INLINE
      void load(element_type const *ptr){
        m_length = s_num_elem;
        for(size_t i = 0;i < s_num_registers;++ i){
          m_registers[i].load(ptr + i*s_num_register_elem);
        }
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
        m_length = s_num_elem;
        for(size_t i = 0;i < s_num_registers;++ i){
          m_registers[i].load(ptr + i*s_num_register_elem*stride, stride);
        }
      }


      /*!
       * @brief Load constructor, assuming scalars are in consecutive memory
       * locations.
       */
      void load_n(element_type const *ptr, size_t len){
        if(len == s_num_elem){
          load(ptr);
        }
        else{
          m_length = len;
          for(size_t i = 0;i < len;++ i){
            set(i, ptr[i]);
          }
        }
      }

      /*!
       * @brief Strided load constructor, when scalars are located in memory
       * locations ptr, ptr+stride, ptr+2*stride, etc.
       *
       *
       * Note: this could be done with "gather" instructions if they are
       * available. (like in avx2, but not in avx)
       */
      void load_n(element_type const *ptr, size_t len, size_t stride){
        if(len == s_num_elem){
          load(ptr, stride);
        }
        else{
          m_length = len;
          for(size_t i = 0;i < len;++ i){
            set(i, ptr[i*stride]);
          }
        }
      }


      /*!
       * @brief Store operation, assuming scalars are in consecutive memory
       * locations.
       */
      void store(element_type *ptr) const{
        if(m_length == s_num_elem){
          for(size_t i = 0;i < s_num_registers;++ i){
            m_registers[i].store(ptr + i*s_num_register_elem);
          }
        }
        else{
          for(size_t i = 0;i < m_length;++ i){
            ptr[i] = (*this)[i];
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
      void store(element_type *ptr, size_t stride) const{
        if(m_length == s_num_elem){
          for(size_t i = 0;i < s_num_registers;++ i){
            m_registers[i].store(ptr + i*s_num_register_elem*stride, stride);
          }
        }
        else{
          for(size_t i = 0;i < m_length;++ i){
            ptr[i*stride] = (*this)[i];
          }
        }
      }


      /*!
       * @brief Assign one register to antoher
       * @param x Vector to copy
       * @return Value of (*this)
       */
      RAJA_INLINE
      self_type const &operator=(self_type const &x)
      {
        m_registers = x.m_registers;
        m_length = x.m_length;
        return *this;
      }



      /*!
       * @brief Assign one register from a scalar
       * @param x Vector to copy
       * @return Value of (*this)
       */
      RAJA_INLINE
      self_type const &operator=(element_type const &x)
      {
        m_length = s_num_elem;
        for(size_t i = 0;i < s_num_registers;++ i){
          m_registers[i] = x;
        }
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
        self_type result = *this;
        result += x;
        return result;
      }


      /*!
       * @brief Add a vector to this vector
       * @param x Vector to add to this register
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      self_type const &operator+=(self_type const &x)
      {
        for(size_t i = 0;i < s_num_registers;++ i){
          m_registers[i] += x.m_registers[i];
        }
        m_length = std::min(m_length, x.m_length);
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
        self_type result = *this;
        result -= x;
        return result;
      }

      /*!
       * @brief Subtract a vector from this vector
       * @param x Vector to subtract from this register
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      self_type const &operator-=(self_type const &x)
      {
        for(size_t i = 0;i < s_num_registers;++ i){
          m_registers[i] -= x.m_registers[i];
        }
        m_length = std::min(m_length, x.m_length);
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
        self_type result = *this;
        result *= x;
        return result;
      }

      /*!
       * @brief Multiply a vector with this vector
       * @param x Vector to multiple with this register
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      self_type const &operator*=(self_type const &x)
      {
        for(size_t i = 0;i < s_num_registers;++ i){
          m_registers[i] *= x.m_registers[i];
        }
        m_length = std::min(m_length, x.m_length);
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
        self_type result = *this;
        result /= x;
        return result;
      }

      /*!
       * @brief Divide this vector by another vector
       * @param x Vector to divide by
       * @return Value of (*this)+x
       */
      RAJA_INLINE
      self_type const &operator/=(self_type const &x)
      {
        for(size_t i = 0;i < s_num_registers;++ i){
          m_registers[i] /= x.m_registers[i];
        }
        m_length = std::min(m_length, x.m_length);
        return *this;
      }

      /*!
       * @brief Sum the elements of this vector
       * @return Sum of the values of the vectors scalar elements
       */
      RAJA_INLINE
      element_type sum() const
      {
        element_type result = (element_type)0;
        if(m_length == s_num_elem){
          for(size_t i = 0;i < s_num_registers;++ i){
            result += m_registers[i].sum();
          }
        }
        else{
          for(size_t i = 0;i < m_length;++ i){
            result += (*this)[i];
          }
        }
        return result;
      }


      /*!
       * @brief Dot product of two vectors
       * @param x Other vector to dot with this vector
       * @return Value of (*this) dot x
       */
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
      RAJA_INLINE
      element_type max() const
      {
        if(m_length == s_num_elem){
          element_type result = m_registers[0].max();
          for(size_t i = 1;i < s_num_registers;++ i){
            result = std::max(result, m_registers[i].max());
          }
          return result;
        }
        else{
          element_type result = (*this)[0];
          for(size_t i = 0;i < m_length;++ i){
            result = std::max(result, (*this)[i]);
          }
          return result;
        }
      }



      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_INLINE
      element_type min() const
      {
        if(m_length == s_num_elem){
          element_type result = m_registers[0].min();
          for(size_t i = 1;i < s_num_registers;++ i){
            result = std::min(result, m_registers[i].min());
          }
          return result;
        }
        else{
          element_type result = (*this)[0];
          for(size_t i = 0;i < m_length;++ i){
            result = std::min(result, (*this)[i]);
          }
          return result;
        }
      }

  };

}  // namespace RAJA


#endif
