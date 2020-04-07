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

#include "RAJA/pattern/vector/internal/MatrixImpl.hpp"

namespace RAJA
{

//  template<typename T, camp::idx_t ROWS, camp::idx_t COLS, MatrixLayout LAYOUT = MATRIX_ROW_MAJOR, typename REGISTER_POLICY = policy::register_default>

  template<typename T, camp::idx_t ROWS, camp::idx_t COLS, MatrixLayout LAYOUT = MATRIX_ROW_MAJOR, typename REGISTER_POLICY = policy::register_default>
  using FixedMatrix =
      internal::MatrixImpl<
        FixedVector<T, ((LAYOUT==MATRIX_ROW_MAJOR) ? COLS:  ROWS), REGISTER_POLICY>,
        LAYOUT,
        camp::make_idx_seq_t<ROWS>,
        camp::make_idx_seq_t<COLS> >;

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
