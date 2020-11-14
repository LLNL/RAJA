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

#ifndef RAJA_pattern_vector_matrixref_HPP
#define RAJA_pattern_vector_matrixref_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#include <array>

namespace RAJA
{

namespace internal
{


  template<typename MATRIX_TYPE, typename INDEX_TYPE,
           typename POINTER_TYPE, bool ROW_STRIDE_ONE, bool COL_STRIDE_ONE>
  class MatrixRef {
    public:
      using self_type =
          MatrixRef<MATRIX_TYPE, INDEX_TYPE, POINTER_TYPE, ROW_STRIDE_ONE, COL_STRIDE_ONE>;

      using matrix_type = MATRIX_TYPE;
      using index_type = INDEX_TYPE;
      using pointer_type = POINTER_TYPE;

      using element_type = typename MATRIX_TYPE::element_type;

      using register_policy = typename MATRIX_TYPE::register_policy;

    private:
      index_type m_linear_index;
      index_type m_row_length;
      index_type m_col_length;
      pointer_type m_data;
      index_type m_row_stride;
      index_type m_col_stride;

    public:


      /*!
       * @brief Default constructor, zeros register contents
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      MatrixRef() :
      m_linear_index(0),
      m_row_length(0),
      m_col_length(0),
      m_data(),
      m_row_stride(0),
      m_col_stride(0)
      {};

      /*!
       * @brief Constructor
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      MatrixRef(index_type lin_index,
                index_type row_length,
                index_type col_length,
                pointer_type pointer,
                index_type row_stride,
                index_type col_stride) :
      m_linear_index(lin_index),
      m_row_length(row_length),
      m_col_length(col_length),
      m_data(pointer),
      m_row_stride(row_stride),
      m_col_stride(col_stride)
      {}


      /*!
       * @brief Copy constructor
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      MatrixRef(self_type const &c) :
      m_linear_index(c.m_linear_index),
      m_row_length(c.m_row_length),
      m_col_length(c.m_col_length),
      m_data(c.m_data),
      m_row_stride(c.m_row_stride),
      m_col_stride(c.m_col_stride)
          {}


      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      constexpr
      bool is_root() {
        return matrix_type::is_root();
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      element_type *get_pointer() const
      {
        return &m_data[m_linear_index];
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      bool is_full() const {
        return m_row_length == MATRIX_TYPE::s_num_rows &&
               m_col_length == MATRIX_TYPE::s_num_cols;
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      constexpr
      bool is_packed() {
        return (ROW_STRIDE_ONE && MATRIX_TYPE::s_layout == MATRIX_COL_MAJOR) ||
               (COL_STRIDE_ONE && MATRIX_TYPE::s_layout == MATRIX_ROW_MAJOR);
      }

      /*!
       * @brief Set entire vector to a single scalar value
       * @param value Value to set all vector elements to
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      void store(matrix_type value) const
      {
        if(is_packed()){
          if(is_full()){
            value.store_packed(m_data+m_linear_index,
                               m_row_stride, m_col_stride);
          }
          else{
            value.store_packed_nm(m_data+m_linear_index,
                       m_row_stride, m_col_stride,
                       m_row_length, m_col_length);
          }
        }
        else{
          if(is_full()){
            value.store_strided(m_data+m_linear_index,
                       m_row_stride, m_col_stride);
          }
          else{
            value.store_strided_nm(m_data+m_linear_index,
                       m_row_stride, m_col_stride,
                       m_row_length, m_col_length);
          }
        }
      }

      /*!
       * @brief Set entire vector to a single scalar value
       * @param value Value to set all vector elements to
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      matrix_type load() const
      {
        matrix_type value;

        if(is_packed()){
          if(is_full()){
            value.load_packed(m_data+m_linear_index,
                       m_row_stride, m_col_stride);
          }
          else{
            value.load_packed_nm(m_data+m_linear_index,
                       m_row_stride, m_col_stride,
                       m_row_length, m_col_length);
          }
        }
        else{
          if(is_full()){
            value.load_strided(m_data+m_linear_index,
                       m_row_stride, m_col_stride);
          }
          else{
            value.load_strided_nm(m_data+m_linear_index,
                       m_row_stride, m_col_stride,
                       m_row_length, m_col_length);
          }
        }

        return value;
      }



      /*!
       * @brief Automatic conversion to the underlying vector_type.
       *
       * This allows the use of a MatrixRef in an expression, and lets the
       * compiler automatically convert a MatrixRef into a load().
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      operator matrix_type() const {
        return load();
      }


      /*!
       * @brief Set entire vector to a single scalar value
       * @param value Value to set all vector elements to
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator=(matrix_type value)
      {
        store(value);
        return *this;
      }



      /*!
       * @brief Set entire vector to a single scalar value
       * @param value Value to set all vector elements to
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type const &operator=(element_type value)
      {
        matrix_type x = value;
        store(x);
        return *this;
      }


      /*!
       * @brief Add two vector registers
       * @param x Vector to add to this register
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      matrix_type operator+(matrix_type const &x) const
      {
        return load() + x;
      }

      /*!
       * @brief Add a vector to this vector
       * @param x Vector to add to this register
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator+=(matrix_type const &x)
      {
        store(load() + x);
        return *this;
      }

      /*!
       * @brief Add a product of two vectors, resulting in an FMA
       * @param x Vector to add to this register
       * @return Value of (*this)+x
       */
      template<typename matrix_type_right>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator+=(MatrixProductRef<matrix_type, matrix_type_right> const &x)
      {
        using helper_t = MatrixMatrixProductHelper<matrix_type, matrix_type_right>;

        store(helper_t::multiply_accumulate(x.get_left(), x.get_right(), load()));

        return *this;
      }

      /*!
       * @brief Subtract two vector registers
       * @param x Vector to subctract from this register
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      matrix_type operator-(matrix_type const &x) const
      {
        return load() - x;
      }

      /*!
       * @brief Subtract a vector from this vector
       * @param x Vector to subtract from this register
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator-=(matrix_type const &x)
      {
        store(load() - x);
        return *this;
      }


      /*!
       * @brief Multiply two vector registers, element wise
       * @param x Vector to subctract from this register
       * @return Value of (*this)+x
       */
      template<typename MT>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      MatrixProductRef<matrix_type, MT>
      operator*(MT const &x) const
      {
        return MatrixProductRef<matrix_type, MT >(load(), x);
      }

      /*!
       * @brief Multiply two MatrixRefs
       */
      template<typename MT, typename ID,
                typename PT, bool RSO, bool CSO>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      MatrixProductRef<matrix_type, MT >
      operator*(MatrixRef<MT, ID, PT, RSO, CSO> const &x) const
      {
        return MatrixProductRef<matrix_type, MT>(load(), x.load());
      }

      /*!
       * @brief Multiply a matrix with this matrix, and store the result
       *
       * Note that this only works with square matrices.
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator*=(matrix_type const &x)
      {
        store(load() * x);
        return *this;
      }



  };





}  // namespace internal
}  // namespace RAJA


#endif
