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


namespace RAJA
{



namespace internal {





  template<typename MATA, typename MATB>
  struct MatrixMatrixMultiplyHelper;

  template<typename T, typename LAYOUT, typename REGISTER_POLICY>
  struct MatrixMatrixMultiplyHelper<
    SquareMatrixRegister<T, LAYOUT, REGISTER_POLICY>,
    SquareMatrixRegister<T, LAYOUT, REGISTER_POLICY>>
    {
      using matrix_type = SquareMatrixRegister<T, LAYOUT, REGISTER_POLICY>;
      using vector_type = typename matrix_type::vector_type;
      using result_type = matrix_type;

      static constexpr camp::idx_t N = matrix_type::vector_type::s_num_elem;


      RAJA_HOST_DEVICE
      static
      RAJA_INLINE
      void multiply_accumulate(matrix_type const &A, matrix_type const &B, matrix_type &C){
        if(LAYOUT::is_row_major()){
#if defined(RAJA_ENABLE_VECTOR_STATS) && !defined(__CUDA_ARCH__)
          RAJA::tensor_stats::num_matrix_mm_multacc_row_row ++;
#endif
          for(camp::idx_t j = 0;j < N;++ j){
            for(camp::idx_t i = 0;i < N;++ i){
                  C.vec(i) =
                      B.vec(j).multiply_add(
                          A.vec(i).get_and_broadcast(j),
                          C.vec(i));

            }
          }
        }
        else{
#if defined(RAJA_ENABLE_VECTOR_STATS) && !defined(__CUDA_ARCH__)
          RAJA::tensor_stats::num_matrix_mm_multacc_col_col ++;
#endif

          for(camp::idx_t j = 0;j < N;++ j){
            for(camp::idx_t i = 0;i < N;++ i){
                C.vec(i) =
                    A.vec(j).multiply_add(
                        B.vec(i).get_and_broadcast(j),
                        C.vec(i));

            }
          }
        }
      }

      RAJA_HOST_DEVICE
      static
      RAJA_INLINE
      void multiply(matrix_type const &A, matrix_type const &B, matrix_type &C){
        C = matrix_type(0);
        multiply_accumulate(A, B, C);
      }


  };





} // namespace internal
} // namespace RAJA




#endif
