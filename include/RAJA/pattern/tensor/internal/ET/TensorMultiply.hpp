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

#ifndef RAJA_pattern_tensor_ET_TensorMultiply_HPP
#define RAJA_pattern_tensor_ET_TensorMultiply_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#include "RAJA/pattern/tensor/TensorRef.hpp"

#include "RAJA/pattern/tensor/internal/ET/ExpressionTemplateBase.hpp"


namespace RAJA
{

  namespace internal
  {

  namespace ET
  {


    template<typename LHS_TYPE, typename RHS_TYPE>
    class TensorMultiply : public TensorExpressionBase<TensorMultiply<LHS_TYPE, RHS_TYPE>> {
      public:
        using self_type = TensorMultiply<LHS_TYPE, RHS_TYPE>;
        using lhs_type = LHS_TYPE;
        using rhs_type = RHS_TYPE;
        using element_type = typename LHS_TYPE::element_type;
        using index_type = typename LHS_TYPE::index_type;
        using tile_type = typename LHS_TYPE::tile_type;
        using result_type = typename LHS_TYPE::result_type;

        RAJA_INLINE
        RAJA_HOST_DEVICE
        TensorMultiply(lhs_type const &lhs, rhs_type const &rhs) :
        m_lhs{lhs}, m_rhs{rhs}
        {}


        template<typename TILE_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        result_type eval(TILE_TYPE const &tile) const {

//          printf("MMMult: "); tile.print();
//          printf("  LHS:"); m_lhs.print();
//          printf("  RHS:"); m_rhs.print();

          // get tile size from matrix type
          index_type tile_size = result_type::s_dim_elem(0);
          index_type k_size = m_lhs.getDimSize(1);
          // TODO: check that lhs and rhs are compatible
          // m_lhs.getDimSize(1) == m_rhs.getDimSize(0)
          // how do we provide checking for this kind of error?

          // tile over row of lhs and column of rhs
          TILE_TYPE lhs_tile = tile;
          lhs_tile.m_size[1] = tile_size;

          TILE_TYPE rhs_tile = tile;
          rhs_tile.m_size[0] = tile_size;

          result_type x(element_type(0));

//          printf("tile_size=%d, k_size=%d, rhs_begin=%d, lhs_begin=%d\n", (int)tile_size, (int)k_size, (int)tile.m_begin[0], (int)tile.m_begin[1]);
//          printf("tile_size=%d\n", (int)tile_size);

          // Do full tiles in k
          index_type k = 0;
          for(;k+tile_size <= k_size; k+= tile_size){
//            printf("k=%d, full tile\n", (int)k);

            // evaluate both sides of operator
            lhs_tile.m_begin[1] = k;
//            printf("  lhs_tile="); lhs_tile.print();
            result_type lhs = m_lhs.eval(lhs_tile);
//            printf("%s\n", lhs.toString().c_str());


            rhs_tile.m_begin[0] = k;
//            printf("  rhs_tile="); rhs_tile.print();
            result_type rhs = m_rhs.eval(rhs_tile);
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
            result_type lhs = m_lhs.eval(lhs_part_tile);
//            printf("  lhs_tile="); lhs_part_tile.print();
//            printf("%s\n", lhs.toString().c_str());



            auto &rhs_part_tile = make_tensor_tile_partial(rhs_tile);
            rhs_part_tile.m_begin[0] = k;
            rhs_part_tile.m_size[0] = k_size-k;
            result_type rhs = m_rhs.eval(rhs_part_tile);
//            printf("  rhs_tile="); rhs_part_tile.print();
//            printf("%s\n", rhs.toString().c_str());

            // compute product into x of partial tile
            x = lhs.multiply_accumulate(rhs, x);
          }

          return x;
        }



      private:
        lhs_type m_lhs;
        rhs_type m_rhs;
    };


    /*
     * Overload for:    arithmetic * tensorexpression

     */
    template<typename LHS, typename RHS,
      typename std::enable_if<std::is_arithmetic<LHS>::value, bool>::type = true,
      typename std::enable_if<std::is_base_of<TensorExpressionConcreteBase, RHS>::value, bool>::type = true>
    RAJA_INLINE
    RAJA_HOST_DEVICE
    auto operator*(LHS const &lhs, RHS const &rhs) ->
    TensorMultiply<typename NormalizeOperandHelper<LHS>::return_type, RHS>
    {
      return TensorMultiply<typename NormalizeOperandHelper<LHS>::return_type, RHS>(NormalizeOperandHelper<LHS>::normalize(lhs), rhs);
    }

  } // namespace ET

  } // namespace internal

}  // namespace RAJA


#endif
