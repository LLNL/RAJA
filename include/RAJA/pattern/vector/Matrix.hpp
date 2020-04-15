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

#ifndef RAJA_pattern_vector_matrix_HPP
#define RAJA_pattern_vector_matrix_HPP

namespace RAJA
{

  enum MatrixLayout {
    MATRIX_ROW_MAJOR,
    MATRIX_COL_MAJOR
  };

  enum MatrixSizeType
  {
    MATRIX_STREAM,
    MATRIX_FIXED
  };

  template<typename T, camp::idx_t ROWS, camp::idx_t COLS, MatrixLayout LAYOUT, typename REGISTER_POLICY, MatrixSizeType SIZE_TYPE>
  class Matrix;

}//namespace RAJA

#include "RAJA/pattern/vector/internal/MatrixImpl.hpp"

namespace RAJA
{

  /*!
   * Wrapping class for internal::MatrixImpl that hides all of the long camp::idx_seq<...> template stuff from the user.
   */
  template<typename T, camp::idx_t ROWS, camp::idx_t COLS, MatrixLayout LAYOUT, typename REGISTER_POLICY, MatrixSizeType SIZE_TYPE>
  class Matrix : public internal::MatrixImpl<Matrix<T, ROWS, COLS, LAYOUT, REGISTER_POLICY, SIZE_TYPE>, REGISTER_POLICY, T, LAYOUT, camp::make_idx_seq_t<ROWS>, camp::make_idx_seq_t<COLS>, SIZE_TYPE>
  {
    public:
      using self_type = Matrix<T, ROWS, COLS, LAYOUT, REGISTER_POLICY, SIZE_TYPE>;
      using base_type = internal::MatrixImpl<self_type, REGISTER_POLICY, T, LAYOUT, camp::make_idx_seq_t<ROWS>, camp::make_idx_seq_t<COLS>, SIZE_TYPE>;

      RAJA_HOST_DEVICE
      RAJA_INLINE
      Matrix(){}

      RAJA_HOST_DEVICE
      RAJA_INLINE
      Matrix(T c) : base_type(c){}


      RAJA_HOST_DEVICE
      RAJA_INLINE
      Matrix(self_type const &c) : base_type(c){}

      template<typename ... RR>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      Matrix(RR const &... rows) : base_type(rows...){}



  };

  template<typename T, camp::idx_t ROWS, camp::idx_t COLS, MatrixLayout LAYOUT = MATRIX_ROW_MAJOR, typename REGISTER_POLICY = policy::register_default>
  using FixedMatrix =
      Matrix<T, ROWS, COLS, LAYOUT, REGISTER_POLICY, MATRIX_FIXED>;

  template<typename T, camp::idx_t ROWS, camp::idx_t COLS, MatrixLayout LAYOUT = MATRIX_ROW_MAJOR, typename REGISTER_POLICY = policy::register_default>
  using StreamMatrix =
      Matrix<T, ROWS, COLS, LAYOUT, REGISTER_POLICY, MATRIX_STREAM>;

  template<typename MATRIX_TYPE>
  using TransposeMatrix = typename internal::MatrixReshapeHelper<MATRIX_TYPE>::similar_transpose_t;


  /*!
   * Converts a Row index to a Column index
   */
  template<typename IDX, typename MATRIX_TYPE>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  ColIndex<IDX, TransposeMatrix<MATRIX_TYPE>> toColIndex(RowIndex<IDX, MATRIX_TYPE> const &r){
    return ColIndex<IDX, TransposeMatrix<MATRIX_TYPE>>(*r, r.size());
  }

  /*!
   * Converts a Column index to a Row index
   */
  template<typename IDX, typename MATRIX_TYPE>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  RowIndex<IDX, TransposeMatrix<MATRIX_TYPE>> toRowIndex(ColIndex<IDX, MATRIX_TYPE> const &c){
    return RowIndex<IDX, TransposeMatrix<MATRIX_TYPE>>(*c, c.size());
  }

}  // namespace RAJA




#endif
