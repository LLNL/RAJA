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

#ifndef RAJA_pattern_tensor_internal_MatrixMatrixMultiply_HPP
#define RAJA_pattern_tensor_internal_MatrixMatrixMultiply_HPP

#include "camp/camp.hpp"
#include "RAJA/config.hpp"
#include "RAJA/pattern/tensor/MatrixRegister.hpp"

//#define DEBUG_MATRIX_LOAD_STORE


namespace RAJA
{



namespace internal {



  template<typename MATA, typename MATB, typename IDX_SEQ>
  struct MatrixMatrixMultiplyHelperExpanded;


  template<typename T, typename LAYOUT, typename REGISTER_POLICY, camp::idx_t ... VAL_SEQ>
  struct MatrixMatrixMultiplyHelperExpanded<
    MatrixRegister<T, LAYOUT, REGISTER_POLICY>,
    MatrixRegister<T, LAYOUT, REGISTER_POLICY>,
    camp::idx_seq<VAL_SEQ...>>
  {
      using matrix_type = MatrixRegister<T, LAYOUT, REGISTER_POLICY>;
      using vector_type = typename matrix_type::vector_type;
      using result_type = matrix_type;

      template<camp::idx_t J>
      RAJA_HOST_DEVICE
      static
      RAJA_INLINE
      int calc_vec_product(matrix_type &sum, matrix_type const &A, matrix_type const &B){

        camp::sink(
                (sum.vec(VAL_SEQ) =
                    B.vec(J).multiply_add(
                        A.vec(VAL_SEQ).get_and_broadcast(J),
                        sum.vec(VAL_SEQ)))...
                );

        return 0;
      }

      RAJA_HOST_DEVICE
      static
      RAJA_INLINE
      matrix_type multiply(matrix_type const &A, matrix_type const &B, matrix_type &C){
        C = matrix_type(0);

        if(LAYOUT::is_row_major()){
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_matrix_mm_mult_row_row ++;
#endif
          camp::sink(calc_vec_product<VAL_SEQ>(C, A, B)...);
        }
        else{
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_matrix_mm_mult_col_col ++;
#endif
          camp::sink(calc_vec_product<VAL_SEQ>(C, B, A)...);
        }
      }

      RAJA_HOST_DEVICE
      static
      RAJA_INLINE
      void multiply_accumulate(matrix_type const &A, matrix_type const &B, matrix_type &C){
        if(LAYOUT::is_row_major()){
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_matrix_mm_multacc_row_row ++;
#endif
          camp::sink(calc_vec_product<VAL_SEQ>(C, A, B)...);
        }
        else{
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_matrix_mm_multacc_col_col ++;
#endif
          camp::sink(calc_vec_product<VAL_SEQ>(C, B, A)...);
        }
      }

  };






  template<typename MATA, typename MATB>
  struct MatrixMatrixMultiplyHelper;

  template<typename T, typename LAYOUT, typename REGISTER_POLICY>
  struct MatrixMatrixMultiplyHelper<
    MatrixRegister<T, LAYOUT, REGISTER_POLICY>,
    MatrixRegister<T, LAYOUT, REGISTER_POLICY>> :
  public
     MatrixMatrixMultiplyHelperExpanded<MatrixRegister<T, LAYOUT, REGISTER_POLICY>,
                                        MatrixRegister<T, LAYOUT, REGISTER_POLICY>,
                                        camp::make_idx_seq_t<MatrixRegister<T, LAYOUT, REGISTER_POLICY>::vector_type::s_num_elem> >
    {};





} // namespace internal
} // namespace RAJA




#endif
