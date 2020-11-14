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


  template<typename T, MatrixLayout LAYOUT, typename REGISTER_POLICY>
  class Matrix;

}//namespace RAJA

#include "RAJA/pattern/vector/internal/MatrixImpl.hpp"

namespace RAJA
{

  /*!
   * Wrapping class for internal::MatrixImpl that hides all of the long camp::idx_seq<...> template stuff from the user.
   */
  template<typename T, MatrixLayout LAYOUT, typename REGISTER_POLICY = RAJA::policy::register_default>
  class Matrix : public internal::MatrixImpl<
    Matrix<T, LAYOUT, REGISTER_POLICY>,
    REGISTER_POLICY,
    T,
    LAYOUT,
    camp::make_idx_seq_t<Register<REGISTER_POLICY, T>::s_num_elem>>
  {
    public:
      using self_type = Matrix<T, LAYOUT, REGISTER_POLICY>;
      using base_type = internal::MatrixImpl<self_type, REGISTER_POLICY, T,
          LAYOUT,
          camp::make_idx_seq_t<Register<REGISTER_POLICY, T>::s_num_elem>>;

      RAJA_HOST_DEVICE
      RAJA_INLINE
      Matrix(){}

      RAJA_HOST_DEVICE
      RAJA_INLINE
      Matrix(T c) : base_type(c){}


      RAJA_HOST_DEVICE
      RAJA_INLINE
      Matrix(self_type const &c) : base_type(c){}


      self_type &operator=(self_type const &c) = default;

      template<typename ... RR>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      Matrix(RR const &... rows) : base_type(rows...){}



  };



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
