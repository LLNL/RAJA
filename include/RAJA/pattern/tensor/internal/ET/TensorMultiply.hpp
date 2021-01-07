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

#include "RAJA/pattern/tensor/internal/ET/ExpressionTemplateBase.hpp"
#include "RAJA/pattern/tensor/internal/ET/DefaultMultiply.hpp"


namespace RAJA
{

  namespace internal
  {

  namespace ET
  {

    // forward decl for FMA contraction
    template<typename LHS_TYPE, typename RHS_TYPE, typename ADD_TYPE>
    class TensorMultiplyAdd;


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

        static constexpr camp::idx_t s_num_dims = result_type::s_num_dims;

        using default_multiply = DefaultMultiply<LHS_TYPE, RHS_TYPE>;

        RAJA_INLINE
        RAJA_HOST_DEVICE
        TensorMultiply(lhs_type const &lhs, rhs_type const &rhs) :
        m_lhs{lhs}, m_rhs{rhs}
        {}


        template<typename TILE_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        result_type eval(TILE_TYPE const &tile) const {
          return default_multiply::multiply(tile, m_lhs, m_rhs);
        }

        /*!
         * Returns the LHS of the operation, used to form contractions
         */
        RAJA_INLINE
        RAJA_HOST_DEVICE
        constexpr
        lhs_type const &getLHS() const {
          return m_lhs;
        }

        /*!
         * Returns the RHS of the operation, used to form contractions
         */
        RAJA_INLINE
        RAJA_HOST_DEVICE
        constexpr
        rhs_type const &getRHS() const {
          return m_rhs;
        }


        /*!
         * operator+ overload that forms a FMA contraction
         */
        template<typename ADD>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        TensorMultiplyAdd<lhs_type, rhs_type, normalize_operand_t<ADD>>
        operator+(ADD const &add) const {
          return TensorMultiplyAdd<lhs_type, rhs_type, normalize_operand_t<ADD>>(m_lhs, m_rhs, normalizeOperand(add));
        }


        RAJA_INLINE
        RAJA_HOST_DEVICE
        void print_ast() const {
          printf("Multiply[");
          default_multiply::print_ast();
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
