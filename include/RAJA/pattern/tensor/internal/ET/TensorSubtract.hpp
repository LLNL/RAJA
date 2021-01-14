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

#ifndef RAJA_pattern_tensor_ET_TensorSubtract_HPP
#define RAJA_pattern_tensor_ET_TensorSubtract_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#include "RAJA/pattern/tensor/internal/ET/ExpressionTemplateBase.hpp"
#include "RAJA/pattern/tensor/internal/ET/Operator.hpp"


namespace RAJA
{

  namespace internal
  {

  namespace ET
  {

    template<typename LHS_TYPE, typename RHS_TYPE>
    class TensorSubtract :  public TensorExpressionBase<TensorAdd<LHS_TYPE, RHS_TYPE>> {
      public:
        using self_type = TensorSubtract<LHS_TYPE, RHS_TYPE>;
        using lhs_type = LHS_TYPE;
        using rhs_type = RHS_TYPE;
        using element_type = typename LHS_TYPE::element_type;
        using index_type = typename LHS_TYPE::index_type;

        using operator_type = Operator<LHS_TYPE, RHS_TYPE>;
        using result_type = typename operator_type::result_type;
        using tile_type = typename operator_type::tile_type;
        static constexpr camp::idx_t s_num_dims = operator_type::s_num_dims;

        RAJA_INLINE
        RAJA_HOST_DEVICE
        TensorSubtract(lhs_type const &lhs, rhs_type const &rhs) :
        m_lhs{lhs}, m_rhs{rhs}
        {}

        RAJA_INLINE
        RAJA_HOST_DEVICE
        constexpr
        index_type getDimSize(index_type dim) const {
          return operator_type::getDimSize(dim, m_lhs, m_rhs);
        }

        template<typename STORAGE, typename TILE_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        void eval(STORAGE &result, TILE_TYPE const &tile) const {
          operator_type::subtract(result, tile, m_lhs, m_rhs);
        }

        RAJA_INLINE
        RAJA_HOST_DEVICE
        void print_ast() const {
          printf("Subtract[");
          operator_type::print_ast();
          printf("](");
          m_lhs.print_ast();
          printf(", ");
          m_rhs.print_ast();
          printf(")");
        }

      private:
        lhs_type m_lhs;
        rhs_type m_rhs;
    };

    /*
     * Overload for:    arithmetic - tensorexpression

     */
    template<typename LHS, typename RHS,
      typename std::enable_if<std::is_arithmetic<LHS>::value, bool>::type = true,
      typename std::enable_if<std::is_base_of<TensorExpressionConcreteBase, RHS>::value, bool>::type = true>
    RAJA_INLINE
    RAJA_HOST_DEVICE
    auto operator-(LHS const &lhs, RHS const &rhs) ->
    TensorSubtract<typename NormalizeOperandHelper<LHS>::return_type, RHS>
    {
      return TensorSubtract<typename NormalizeOperandHelper<LHS>::return_type, RHS>(NormalizeOperandHelper<LHS>::normalize(lhs), rhs);
    }


  } // namespace ET

  } // namespace internal

}  // namespace RAJA


#endif
