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

#ifndef RAJA_pattern_vector_fixedvector_HPP
#define RAJA_pattern_vector_fixedvector_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#include <array>

namespace RAJA
{


/*!
 * \file
 * Vector operation functions in the namespace RAJA

 *
 */

  template<typename REGISTER_TYPE, size_t NUM_ELEM>
  class FixedVector;

  template<template<typename, typename, size_t> class REGISTER_TYPE, typename REGISTER_POLICY, typename ELEMENT_TYPE, size_t NUM_REG_ELEM, size_t NUM_ELEM>
  class FixedVector<REGISTER_TYPE<REGISTER_POLICY, ELEMENT_TYPE, NUM_REG_ELEM>, NUM_ELEM>
  {
    public:
      using full_register_type =
          REGISTER_TYPE<REGISTER_POLICY, ELEMENT_TYPE, NUM_REG_ELEM>;
      static constexpr size_t s_num_register_elem = NUM_REG_ELEM;

      using self_type = FixedVector<full_register_type, NUM_ELEM>;
      using element_type = ELEMENT_TYPE;


      static constexpr size_t s_num_elem = NUM_ELEM;
      static constexpr size_t s_byte_width = sizeof(element_type);
      static constexpr size_t s_bit_width = s_byte_width*8;


      static constexpr size_t s_num_full_registers = s_num_elem / s_num_register_elem;

      static constexpr size_t s_num_full_elem = s_num_full_registers*s_num_register_elem;

      static constexpr size_t s_num_partial_registers =
          s_num_full_elem == s_num_elem ? 0 : 1;

      static constexpr size_t s_num_partial_elem = s_num_elem - s_num_full_elem;

      using partial_register_type =
          REGISTER_TYPE<REGISTER_POLICY, ELEMENT_TYPE, s_num_partial_elem ? s_num_partial_elem : 1>;

    private:
      std::array<full_register_type, s_num_full_registers> m_full_registers;
      std::array<partial_register_type, s_num_partial_registers> m_partial_register;
    public:


      /*!
       * @brief Default constructor, zeros register contents
       */
      FixedVector() = default;

      /*!
       * @brief Copy constructor
       */
      RAJA_INLINE
      FixedVector(self_type const &c) :
        m_full_registers(c.m_full_registers),
        m_partial_register(c.m_partial_register)
          {}

      /*!
       * @brief Scalar constructor (broadcast)
       */
      RAJA_INLINE
      FixedVector(element_type const &c)
      {
        for(size_t i = 0;i < s_num_full_registers;++ i){
          m_full_registers[i] = c;
        }
        if(s_num_partial_registers){
          m_partial_register[0] = c;
        }
      }


      /*!
       * @brief Load constructor, assuming scalars are in consecutive memory
       * locations.
       */
      RAJA_INLINE
      void load(element_type const *ptr){
        for(size_t i = 0;i < s_num_full_registers;++ i){
          m_full_registers[i].load(ptr + i*s_num_register_elem);
        }
        if(s_num_partial_registers){
          m_partial_register[0].load(ptr + s_num_full_elem);
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
        for(size_t i = 0;i < s_num_full_registers;++ i){
          m_full_registers[i].load(ptr + i*stride*s_num_register_elem, stride);
        }
        if(s_num_partial_registers){
          m_partial_register[0].load(ptr + stride*s_num_full_elem, stride);
        }
      }


      /*!
       * @brief Store operation, assuming scalars are in consecutive memory
       * locations.
       */
      RAJA_INLINE
      void store(element_type *ptr) const{
        for(size_t i = 0;i < s_num_full_registers;++ i){
          m_full_registers[i].store(ptr + i*s_num_register_elem);
        }
        if(s_num_partial_registers){
          m_partial_register[0].store(ptr + s_num_full_elem);
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
      void store(element_type *ptr, size_t stride) const{
        for(size_t i = 0;i < s_num_full_registers;++ i){
          m_full_registers[i].store(ptr + i*stride*s_num_register_elem, stride);
        }
        if(s_num_partial_registers){
          m_partial_register[0].store(ptr + stride*s_num_full_elem, stride);
        }
      }


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

        if(r < s_num_full_registers){
          return m_full_registers[r][e];
        }
        return m_partial_register[0][e];
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

        if(r < s_num_full_registers){
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
      RAJA_INLINE
      self_type const &operator=(element_type value)
      {
        for(size_t i = 0;i < s_num_full_registers;++ i){
          m_full_registers[i] = value;
        }
        if(s_num_partial_registers){
          m_partial_register[0] = value;
        }
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
        for(size_t i = 0;i < s_num_full_registers;++ i){
          m_full_registers[i] = x.m_full_registers[i];
        }
        if(s_num_partial_registers){
          m_partial_register[0] = x.m_partial_register[0];
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
        self_type result(*this);

        for(size_t i = 0;i < s_num_full_registers;++ i){
          result.m_full_registers[i] += x.m_full_registers[i];
        }
        if(s_num_partial_registers){
          result.m_partial_register[0] += x.m_partial_register[0];
        }

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
        for(size_t i = 0;i < s_num_full_registers;++ i){
          m_full_registers[i] += x.m_full_registers[i];
        }
        if(s_num_partial_registers){
          m_partial_register[0] += x.m_partial_register[0];
        }

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
        self_type result(*this);

        for(size_t i = 0;i < s_num_full_registers;++ i){
          result.m_full_registers[i] -= x.m_full_registers[i];
        }
        if(s_num_partial_registers){
          result.m_partial_register[0] -= x.m_partial_register[0];
        }

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
        for(size_t i = 0;i < s_num_full_registers;++ i){
          m_full_registers[i] -= x.m_full_registers[i];
        }
        if(s_num_partial_registers){
          m_partial_register[0] -= x.m_partial_register[0];
        }

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
        self_type result(*this);

        for(size_t i = 0;i < s_num_full_registers;++ i){
          result.m_full_registers[i] *= x.m_full_registers[i];
        }
        if(s_num_partial_registers){
          result.m_partial_register[0] *= x.m_partial_register[0];
        }

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
        for(size_t i = 0;i < s_num_full_registers;++ i){
          m_full_registers[i] *= x.m_full_registers[i];
        }
        if(s_num_partial_registers){
          m_partial_register[0] *= x.m_partial_register[0];
        }

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
        self_type result(*this);

        for(size_t i = 0;i < s_num_full_registers;++ i){
          result.m_full_registers[i] /= x.m_full_registers[i];
        }
        if(s_num_partial_registers){
          result.m_partial_register[0] /= x.m_partial_register[0];
        }

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
        for(size_t i = 0;i < s_num_full_registers;++ i){
          m_full_registers[i] /= x.m_full_registers[i];
        }
        if(s_num_partial_registers){
          m_partial_register[0] /= x.m_partial_register[0];
        }

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
        for(size_t i = 0;i < s_num_full_registers;++ i){
          result += m_full_registers[i].sum();
        }
        if(s_num_partial_registers){
          result += m_partial_register[0].sum();
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
        element_type result = (element_type)0;
        for(size_t i = 0;i < s_num_full_registers;++ i){
          result += m_full_registers[i].dot(x.m_full_registers[i]);
        }
        if(s_num_partial_registers){
          result += m_partial_register[0].dot(x.m_partial_register[0]);
        }
        return result;
      }


      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_INLINE
      element_type max() const
      {
        if(s_num_full_registers == 0){
          return m_partial_register[0].max();
        }

        element_type result = (element_type)m_full_registers[0].max();
        for(size_t i = 1;i < s_num_full_registers;++ i){
          result = std::max<double>(result, m_full_registers[i].max());
        }
        if(s_num_partial_registers){
          result = std::max<double>(result, m_partial_register[0].max());
        }
        return result;
      }

      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_INLINE
      element_type min() const
      {
        if(s_num_full_registers == 0){
          return m_partial_register[0].min();
        }

        element_type result = (element_type)m_full_registers[0].min();
        for(size_t i = 1;i < s_num_full_registers;++ i){
          result = std::min<double>(result, m_full_registers[i].min());
        }
        if(s_num_partial_registers){
          result = std::min<double>(result, m_partial_register[0].min());
        }
        return result;
      }

  };



  template<typename ST, typename REGISTER_TYPE, size_t NUM_ELEM>
  FixedVector<REGISTER_TYPE, NUM_ELEM>
  operator+(ST x, FixedVector<REGISTER_TYPE, NUM_ELEM> const &y){
    return FixedVector<REGISTER_TYPE, NUM_ELEM>(x) + y;
  }

  template<typename ST, typename REGISTER_TYPE, size_t NUM_ELEM>
  FixedVector<REGISTER_TYPE, NUM_ELEM>
  operator-(ST x, FixedVector<REGISTER_TYPE, NUM_ELEM> const &y){
    return FixedVector<REGISTER_TYPE, NUM_ELEM>(x) - y;
  }

  template<typename ST, typename REGISTER_TYPE, size_t NUM_ELEM>
  FixedVector<REGISTER_TYPE, NUM_ELEM>
  operator*(ST x, FixedVector<REGISTER_TYPE, NUM_ELEM> const &y){
    return FixedVector<REGISTER_TYPE, NUM_ELEM>(x) * y;
  }

  template<typename ST, typename REGISTER_TYPE, size_t NUM_ELEM>
  FixedVector<REGISTER_TYPE, NUM_ELEM>
  operator/(ST x, FixedVector<REGISTER_TYPE, NUM_ELEM> const &y){
    return FixedVector<REGISTER_TYPE, NUM_ELEM>(x) / y;
  }

}  // namespace RAJA


#endif
