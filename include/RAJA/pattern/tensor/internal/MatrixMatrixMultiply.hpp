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

  template<typename T, typename LAYOUT, typename REGISTER_POLICY, camp::idx_t N_SIZE, camp::idx_t M_SIZE, camp::idx_t M2_SIZE, camp::idx_t O_SIZE>
  struct MatrixMatrixMultiplyHelper<
    TensorRegister<REGISTER_POLICY,
                   T,
                   LAYOUT,
                   camp::idx_seq<N_SIZE, M_SIZE>>,
     TensorRegister<REGISTER_POLICY,
                    T,
                    LAYOUT,
                    camp::idx_seq<M2_SIZE, O_SIZE>> >
    {

      static_assert(M_SIZE == M2_SIZE, "Matrices are not compatible for multiplication");

      using left_type = TensorRegister<REGISTER_POLICY,
                                       T,
                                       LAYOUT,
                                       camp::idx_seq<N_SIZE, M_SIZE>>;

      using right_type = TensorRegister<REGISTER_POLICY,
                                        T,
                                        LAYOUT,
                                        camp::idx_seq<M_SIZE, O_SIZE>> ;

      using result_type = TensorRegister<REGISTER_POLICY,
                                         T,
                                         LAYOUT,
                                         camp::idx_seq<N_SIZE, O_SIZE>> ;

      using register_type = typename result_type::register_type;

      static constexpr camp::idx_t s_elements_per_register = left_type::s_elements_per_register;
      static constexpr camp::idx_t s_A_minor_dim_registers = left_type::s_minor_dim_registers;
      static constexpr camp::idx_t s_B_minor_dim_registers = right_type::s_minor_dim_registers;
      static constexpr camp::idx_t s_C_minor_dim_registers = result_type::s_minor_dim_registers;


      static_assert( (LAYOUT::is_row_major()    && (s_B_minor_dim_registers==s_C_minor_dim_registers)) ||
                     (LAYOUT::is_column_major() && (s_A_minor_dim_registers==s_C_minor_dim_registers)),
                     "Result matrix is incompatible with operands");

      RAJA_HOST_DEVICE
      static
      RAJA_INLINE
      void multiply_accumulate(left_type const &A, right_type const &B, result_type &C){
        if(LAYOUT::is_row_major()){
#if defined(RAJA_ENABLE_VECTOR_STATS) && !defined(__CUDA_ARCH__)
          RAJA::tensor_stats::num_matrix_mm_multacc_row_row ++;
#endif

          // Matrix A has 1 more more registers per row
          if(s_A_minor_dim_registers > 0){
            // rows in A
            camp::idx_t regA = 0;
            camp::idx_t regC = 0;

            for(camp::idx_t i = 0;i < N_SIZE;++ i){


              // registers in row of A (registers in j)
              camp::idx_t regB = 0;
              for(camp::idx_t rowregA = 0;rowregA < s_A_minor_dim_registers;++ rowregA){

                // columns within regA
                for(camp::idx_t j = 0;j < s_elements_per_register;++ j){

                  // get value of regA to apply to row of B
                  register_type Atmp = A.get_register(regA).get_and_broadcast(j);

                  // registers in row of
                  for(camp::idx_t rowregBC = 0;rowregBC < s_B_minor_dim_registers;++ rowregBC){

//                    printf("i=%d, j=%d, rowregA=%d, rowregBC=%d, regA=%d, regB=%d, regC=%d\n",
//                        (int)i, (int)j, (int)rowregA, (int)rowregBC, (int)regA, (int)regB, (int)regC);

                      C.get_register(regC+rowregBC) =
                          B.get_register(regB).multiply_add(
                              Atmp,
                              C.get_register(regC+rowregBC));

                      ++ regB;
                  } // rowregBC

                } // j

                ++ regA;
              }// rowregA

              regC += s_C_minor_dim_registers;
            } // i

          }
        }
        else{
#if defined(RAJA_ENABLE_VECTOR_STATS) && !defined(__CUDA_ARCH__)
          RAJA::tensor_stats::num_matrix_mm_multacc_col_col ++;
#endif

          for(camp::idx_t i = 0;i < O_SIZE;++ i){
            for(camp::idx_t j = 0;j < M_SIZE;++ j){
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
      void multiply(left_type const &A, right_type const &B, result_type &C){
        C = result_type(0);
        multiply_accumulate(A, B, C);
      }


  };





} // namespace internal
} // namespace RAJA




#endif
