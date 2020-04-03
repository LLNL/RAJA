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

  template<typename VECTOR_TYPE, MatrixLayout LAYOUT, camp::idx_t ... IDX_REG, camp::idx_t ... IDX_ROW, camp::idx_t ... IDX_COL>
  class MatrixBase<MatrixImpl<VECTOR_TYPE, LAYOUT, camp::idx_seq<IDX_REG...>, camp::idx_seq<IDX_ROW...>, camp::idx_seq<IDX_COL...> >>
  {
    public:
      using self_type = MatrixImpl<VECTOR_TYPE, LAYOUT, camp::idx_seq<IDX_REG...>, camp::idx_seq<IDX_ROW...>, camp::idx_seq<IDX_COL...> >;

      using vector_type = VECTOR_TYPE;
      using row_vector_type = changeVectorLength<VECTOR_TYPE, sizeof...(IDX_COL)>;
      using col_vector_type = changeVectorLength<VECTOR_TYPE, sizeof...(IDX_ROW)>;
      using element_type = typename VECTOR_TYPE::element_type;


      static constexpr camp::idx_t s_num_rows = sizeof...(IDX_ROW);
      static constexpr camp::idx_t s_num_cols = sizeof...(IDX_COL);

      RAJA_INLINE
      static constexpr camp::idx_t num_elem(camp::idx_t dim){
        return (dim==0) ? s_num_rows : s_num_cols;
      }

    private:
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type *getThis(){
        return static_cast<self_type *>(this);
      }

      RAJA_INLINE
      RAJA_HOST_DEVICE
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
        return VECTOR_TYPE::is_root();
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
      template<typename VT, MatrixLayout L, typename REG, typename ROW, typename COL>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      typename MatrixMatrixProductHelper<self_type, MatrixImpl<VT, L, REG, ROW, COL>>::result_type
      operator*(MatrixImpl<VT, L, REG, ROW, COL> const &mat) const {
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





#if 0
      /*!
       * @brief Fused multiply add: fma(b, c) = (*this)*b+c
       *
       * Derived types can override this to implement intrinsic FMA's
       *
       * @param b Second product operand
       * @param c Sum operand
       * @return Value of (*this)*b+c
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type fused_multiply_add(self_type const &b, self_type const &c) const
      {
        return (self_type(*getThis()) * self_type(b)) + self_type(c);
      }

      /*!
       * @brief Fused multiply subtract: fms(b, c) = (*this)*b-c
       *
       * Derived types can override this to implement intrinsic FMS's
       *
       * @param b Second product operand
       * @param c Subtraction operand
       * @return Value of (*this)*b-c
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type fused_multiply_subtract(self_type const &b, self_type const &c) const
      {
        return getThis()->fused_multiply_add(b, -c);
      }
#endif
  };

  }
}  // namespace RAJA



#endif
