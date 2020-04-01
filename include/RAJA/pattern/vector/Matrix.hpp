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

  template<typename T, camp::idx_t ROWS, camp::idx_t COLS, typename REGISTER_POLICY  = policy::register_default>
  using FixedMatrix =
      internal::MatrixImpl<FixedVector<T, COLS, REGISTER_POLICY>, MATRIX_ROW_MAJOR, camp::make_idx_seq_t<FixedVector<T, COLS, REGISTER_POLICY>::num_registers()>, camp::make_idx_seq_t<ROWS>, camp::make_idx_seq_t<COLS> >;

  template<typename MATRIX_TYPE>
  using TransposeMatrix = typename internal::MatrixReshapeHelper<MATRIX_TYPE>::similar_transpose_t;

}  // namespace RAJA




#endif
