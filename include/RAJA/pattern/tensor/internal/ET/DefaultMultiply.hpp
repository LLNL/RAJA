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

#ifndef RAJA_pattern_tensor_ET_DefaultMultiply_HPP
#define RAJA_pattern_tensor_ET_DefaultMultiply_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#include "RAJA/pattern/tensor/internal/ET/ExpressionTemplateBase.hpp"


namespace RAJA
{

  namespace internal
  {

  namespace ET
  {


    /*!
     * Provides default multiply, multiply add, and multiply subtract
     * operations.
     *
     * If the operands are both matrices, we perform a matrix-matrix multiply.
     * Otherwise, we perform element-wise operations.
     */
    template<typename LHS_TYPE, typename RHS_TYPE, class ENABLE = void>
    struct DefaultMultiply{

        using result_type = typename LHS_TYPE::result_type;


        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        void print_ast() {
          printf("Element-Wise");
        }

        /*!
         * Evaluate operands and perform element-wise multiply
         */
        template<typename TILE_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        result_type multiply(TILE_TYPE const &tile, LHS_TYPE const &lhs, RHS_TYPE const &rhs){
          return lhs.eval(tile).multiply(rhs.eval(tile));
        }


        /*!
         * Evaluate operands and perform element-wise multiply add
         */
        template<typename TILE_TYPE, typename ADD_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        result_type multiply_add(TILE_TYPE const &tile, LHS_TYPE const &lhs, RHS_TYPE const &rhs, ADD_TYPE const &add){
          return lhs.eval(tile).multiply_add(rhs.eval(tile), add.eval(tile));
        }


        /*!
         * Evaluate operands and perform element-wise multiply subtract
         */
        template<typename TILE_TYPE, typename SUB_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        result_type multiply_subtract(TILE_TYPE const &tile, LHS_TYPE const &lhs, RHS_TYPE const &rhs, SUB_TYPE const &sub){
          return lhs.eval(tile).multiply_subtract(rhs.eval(tile), sub.eval(tile));
        }


    };


    /*!
     * Specialization for matrix-matrix multiplication.
     *
     * By default the A*B operator for two matrices produces a matrix-matrix
     * multiplication.
     *
     */
    template<typename LHS_TYPE, typename RHS_TYPE>
    struct DefaultMultiply<LHS_TYPE, RHS_TYPE,
    typename std::enable_if<LHS_TYPE::s_num_dims == 2 && RHS_TYPE::s_num_dims==2>::type>
    {

      using lhs_type = LHS_TYPE;
      using rhs_type = RHS_TYPE;
      using element_type = typename LHS_TYPE::element_type;
      using index_type = typename LHS_TYPE::index_type;
      using tile_type = typename LHS_TYPE::tile_type;
      using result_type = typename LHS_TYPE::result_type;

      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      void print_ast() {
        printf("Matrx-Matrix");
      }

      /*!
       * Evaluate operands and perform element-wise multiply
       */
      template<typename TILE_TYPE>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      result_type multiply(TILE_TYPE const &tile, LHS_TYPE const &lhs, RHS_TYPE const &rhs){
        return multiply_add(tile, lhs, rhs, result_type(element_type(0)) );
      }

      template<typename TILE_TYPE, typename ADD_TYPE>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      result_type multiply_add(TILE_TYPE const &tile, LHS_TYPE const &et_lhs, RHS_TYPE const &et_rhs, ADD_TYPE const &add){


//          printf("MMMult: "); tile.print();
//          printf("  LHS:"); m_lhs.print();
//          printf("  RHS:"); m_rhs.print();

        // get tile size from matrix type
        index_type tile_size = result_type::s_dim_elem(0);
        index_type k_size = et_lhs.getDimSize(1);
        // TODO: check that lhs and rhs are compatible
        // m_lhs.getDimSize(1) == m_rhs.getDimSize(0)
        // how do we provide checking for this kind of error?

        // tile over row of lhs and column of rhs
        TILE_TYPE lhs_tile = tile;
        lhs_tile.m_size[1] = tile_size;

        TILE_TYPE rhs_tile = tile;
        rhs_tile.m_size[0] = tile_size;

        // start with value of m_add  (THIS IS THE ONLY DIIF to TensorMultiply)
        result_type x(add);

//          printf("tile_size=%d, k_size=%d, rhs_begin=%d, lhs_begin=%d\n", (int)tile_size, (int)k_size, (int)tile.m_begin[0], (int)tile.m_begin[1]);
//          printf("tile_size=%d\n", (int)tile_size);

        // Do full tiles in k
        index_type k = 0;
        for(;k+tile_size <= k_size; k+= tile_size){
//            printf("k=%d, full tile\n", (int)k);

          // evaluate both sides of operator
          lhs_tile.m_begin[1] = k;
//            printf("  lhs_tile="); lhs_tile.print();
          result_type lhs = et_lhs.eval(lhs_tile);
//            printf("%s\n", lhs.toString().c_str());


          rhs_tile.m_begin[0] = k;
//            printf("  rhs_tile="); rhs_tile.print();
          result_type rhs = et_rhs.eval(rhs_tile);
//            printf("%s\n", rhs.toString().c_str());


          // compute product into x
          x = lhs.multiply_accumulate(rhs, x);
        }
        // remainder tile in k
        if(k < k_size){
//            printf("k=%d, partial tile\n", (int)k);
          auto &lhs_part_tile = make_tensor_tile_partial(lhs_tile);
          lhs_part_tile.m_begin[1] = k;
          lhs_part_tile.m_size[1] = k_size-k;
          result_type lhs = et_lhs.eval(lhs_part_tile);
//            printf("  lhs_tile="); lhs_part_tile.print();
//            printf("%s\n", lhs.toString().c_str());



          auto &rhs_part_tile = make_tensor_tile_partial(rhs_tile);
          rhs_part_tile.m_begin[0] = k;
          rhs_part_tile.m_size[0] = k_size-k;
          result_type rhs = et_rhs.eval(rhs_part_tile);
//            printf("  rhs_tile="); rhs_part_tile.print();
//            printf("%s\n", rhs.toString().c_str());

          // compute product into x of partial tile
          x = lhs.multiply_accumulate(rhs, x);
        }

        return x;
      }

    };




  } // namespace ET

  } // namespace internal

}  // namespace RAJA


#endif
