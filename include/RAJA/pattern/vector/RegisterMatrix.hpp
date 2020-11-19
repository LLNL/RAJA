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

  namespace internal{
    template<typename REGISTER_POLICY, typename ELEMENT_TYPE, typename LAYOUT, typename IDX_SEQ>
    class RegisterMatrixImpl;
  }

  template<typename T, typename LAYOUT, typename REGISTER_POLICY = RAJA::policy::register_default>
  using RegisterMatrix = internal::RegisterMatrixImpl<
      REGISTER_POLICY, T, LAYOUT, camp::make_idx_seq_t<Register<REGISTER_POLICY, T>::s_num_elem> >;



}//namespace RAJA

#include "internal/RegisterMatrixImpl.hpp"

namespace RAJA
{


  /*!
   * Converts a Row index to a Column index
   */
  template<typename IDX, typename MATRIX_TYPE>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  constexpr
  ColIndex<IDX, MATRIX_TYPE> toColIndex(RowIndex<IDX, MATRIX_TYPE> const &r){
    return ColIndex<IDX, MATRIX_TYPE>(*r, r.size());
  }

  /*!
   * Converts a Column index to a Row index
   */
  template<typename IDX, typename MATRIX_TYPE>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  constexpr
  RowIndex<IDX, MATRIX_TYPE> toRowIndex(ColIndex<IDX, MATRIX_TYPE> const &c){
    return RowIndex<IDX, MATRIX_TYPE>(*c, c.size());
  }

}  // namespace RAJA




#endif
