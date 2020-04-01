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
      internal::MatrixImpl<typename internal::FixedVectorTypeHelper<Register<REGISTER_POLICY, T, 4>, COLS>::type, MATRIX_ROW_MAJOR, camp::make_idx_seq_t<internal::FixedVectorTypeHelper<Register<REGISTER_POLICY, T, 4>, COLS>::s_num_registers>, camp::make_idx_seq_t<ROWS>, camp::make_idx_seq_t<COLS> >;


}  // namespace RAJA




#endif
