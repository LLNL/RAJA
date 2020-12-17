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

#ifndef RAJA_pattern_simd_register_matrix_HPP
#define RAJA_pattern_simd_register_matrix_HPP

#include "RAJA/policy/tensor/arch.hpp"

namespace RAJA
{

  template<camp::idx_t ROW, camp::idx_t COL>
  struct MatrixLayout : public camp::idx_seq<ROW, COL>{
    static_assert(ROW == 0 || COL == 0, "invalid template arguments");
    static_assert(ROW == 1 || COL == 1, "invalid template arguments");
    static_assert(ROW+COL == 1, "invalid template arguments");

    RAJA_INLINE
    RAJA_HOST_DEVICE
    static
    constexpr
    bool is_column_major(){
      return COL == 1;
    }

    RAJA_INLINE
    RAJA_HOST_DEVICE
    static
    constexpr
    bool is_row_major(){
      return ROW == 1;
    }
  };


  using MATRIX_ROW_MAJOR = MatrixLayout<1, 0>;
  using MATRIX_COL_MAJOR = MatrixLayout<0, 1>;

  struct VectorLayout{};

  namespace internal{
    template<typename REGISTER_POLICY, typename ELEMENT_TYPE, typename LAYOUT, typename IDX_SEQ>
    class RegisterMatrixImpl;
  }

  template<typename T, typename LAYOUT, typename REGISTER_POLICY = RAJA::default_register>
  using RegisterMatrix = internal::RegisterMatrixImpl<
      REGISTER_POLICY, T, LAYOUT,
      camp::make_idx_seq_t<RegisterTraits<REGISTER_POLICY, T>::s_num_elem> >;



}//namespace RAJA

#include "RegisterMatrixImpl.hpp"



#endif
