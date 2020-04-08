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

#ifndef RAJA_pattern_vector_matrixbase_HPP
#define RAJA_pattern_vector_matrixbase_HPP

namespace RAJA
{

  namespace internal {
  /*!
   * Matrix base class that provides some default behaviors and functionality
   * that is similar between Matrix specializations (row vs col major, etc.)
   */
  template<typename Derived>
  class MatrixBase;

  template<typename MATRIX_TYPE, typename REGISTER_POLICY, typename ELEMENT_TYPE, MatrixLayout LAYOUT, camp::idx_t ... IDX_ROW, camp::idx_t ... IDX_COL, MatrixSizeType SIZE_TYPE>
  class MatrixBase<MatrixImpl<MATRIX_TYPE, REGISTER_POLICY, ELEMENT_TYPE, LAYOUT, camp::idx_seq<IDX_ROW...>, camp::idx_seq<IDX_COL...>, SIZE_TYPE >>
  {
    public:
      //using self_type = MatrixImpl<MATRIX_TYPE, REGISTER_POLICY, ELEMENT_TYPE, LAYOUT, camp::idx_seq<IDX_ROW...>, camp::idx_seq<IDX_COL...>, SIZE_TYPE >;
      using self_type = MATRIX_TYPE;

      static constexpr VectorSizeType s_vector_size_type = (SIZE_TYPE==MATRIX_FIXED) ? VECTOR_FIXED : VECTOR_STREAM;
      using row_vector_type = Vector<REGISTER_POLICY, ELEMENT_TYPE, sizeof...(IDX_COL), s_vector_size_type>;
      using col_vector_type = Vector<REGISTER_POLICY, ELEMENT_TYPE, sizeof...(IDX_ROW), s_vector_size_type>;

      using element_type = ELEMENT_TYPE;
      using register_policy = REGISTER_POLICY;

      static constexpr MatrixLayout s_layout = LAYOUT;
      static constexpr MatrixSizeType s_size_type = SIZE_TYPE;
      static constexpr camp::idx_t s_num_rows = sizeof...(IDX_ROW);
      static constexpr camp::idx_t s_num_cols = sizeof...(IDX_COL);

    private:
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type *getThis(){
        return static_cast<self_type *>(this);
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      self_type const *getThis() const{
        return static_cast<self_type const *>(this);
      }

    public:

      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      constexpr
      bool is_root() {
        return row_vector_type::is_root();
      }

      /*!
       * Gets the maximum size of matrix along specified dimension
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      constexpr camp::idx_t s_dim_elem(camp::idx_t dim){
        return (dim==0) ? s_num_rows : s_num_cols;
      }


      /*!
       * @brief Set entire vector to a single scalar value
       * @param value Value to set all vector elements to
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator=(element_type value)
      {
        getThis()->broadcast(value);
        return *this;
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
        getThis()->copy(x);
        return *this;
      }

      template<typename IDX_I, typename IDX_J>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type operator()(IDX_I row, IDX_J col){
        return getThis()->get(row, col);
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
        return getThis()->add(x);
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
        *getThis() = getThis()->add(x);
        return *getThis();
      }

      /*!
       * @brief Negate the value of this vector
       * @return Value of -(*this)
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type operator-() const
      {
        return self_type(0).subtract(*getThis());
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
        return getThis()->subtract(x);
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
        *getThis() = getThis()->subtract(x);
        return *getThis();
      }

      /*!
       * Matrix vector product
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      col_vector_type operator*(row_vector_type v) const {
        return getThis()->right_multiply_vector(v);
      }

      /*!
       * @brief Multiply two vector registers, element wise
       * @param x Vector to subctract from this register
       * @return Value of (*this)+x
       */

      template<camp::idx_t ROWS, camp::idx_t COLS>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      typename MatrixMatrixProductHelper<self_type, Matrix<ELEMENT_TYPE, ROWS, COLS, LAYOUT, REGISTER_POLICY, SIZE_TYPE>>::result_type
      operator*(Matrix<ELEMENT_TYPE, ROWS, COLS, LAYOUT, REGISTER_POLICY, SIZE_TYPE> const &mat) const {
        return getThis()->multiply(mat);
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
        *getThis() = getThis()->multiply(x);
        return *getThis();
      }





  };

  }
}  // namespace RAJA



#endif
