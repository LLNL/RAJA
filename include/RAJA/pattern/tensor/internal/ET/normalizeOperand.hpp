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
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_tensor_ET_normalizeOperand_HPP
#define RAJA_pattern_tensor_ET_normalizeOperand_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#include "RAJA/pattern/tensor/internal/TensorRef.hpp"


namespace RAJA
{
namespace internal
{
namespace expt
{


    class TensorRegisterConcreteBase;

  namespace ET
  {
    class TensorExpressionConcreteBase;

    template<typename RHS, typename enable = void>
    struct NormalizeOperandHelper;


    /*
     * For TensorExpression nodes, we just return them as-is.
     */
    template<typename RHS>
    struct NormalizeOperandHelper<RHS,
    typename std::enable_if<std::is_base_of<TensorExpressionConcreteBase, RHS>::value>::type>
    {
        using return_type = RHS;

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        constexpr
        return_type normalize(RHS const &rhs){
          return rhs;
        }
    };




    /**
     * Allows uniform packaging up of operands into ExpressionTemplates.
     *
     * The NormalizeOperandHelper is specialized throughout the code in order
     * to convert non-ET operands into ET objects
     *
     * ET operators can then take any operand type, and use this to convert
     * them into ET types the same way.
     */
    template<typename RHS>
    RAJA_INLINE
    RAJA_HOST_DEVICE
    auto normalizeOperand(RHS const &rhs) ->
    typename NormalizeOperandHelper<RHS>::return_type
    {
      return NormalizeOperandHelper<RHS>::normalize(rhs);
    }

    template<typename RHS>
    using normalize_operand_t =
        typename NormalizeOperandHelper<RHS>::return_type;


  } // namespace ET

  } // namespace internal
} // namespace expt

}  // namespace RAJA


#endif
