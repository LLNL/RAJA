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
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_tensor_ET_TensorNegate_HPP
#define RAJA_pattern_tensor_ET_TensorNegate_HPP

#include "RAJA/config.hpp"
#include "RAJA/pattern/tensor/internal/ET/ExpressionTemplateBase.hpp"
#include "RAJA/util/macros.hpp"


namespace RAJA
{
namespace internal
{
namespace expt
{


namespace ET
{

template <typename ET_TYPE>
class TensorNegate : public TensorExpressionBase<TensorNegate<ET_TYPE>>
{
public:
  using self_type = TensorNegate<ET_TYPE>;
  using rhs_type = ET_TYPE;
  using tensor_type = typename ET_TYPE::result_type;
  using element_type = typename tensor_type::element_type;
  using index_type = typename ET_TYPE::index_type;

  using result_type = tensor_type;
  using tile_type = typename ET_TYPE::tile_type;
  static constexpr camp::idx_t s_num_dims = ET_TYPE::s_num_dims;

  RAJA_INLINE
  RAJA_HOST_DEVICE
  TensorNegate(rhs_type const &tensor) : m_tensor{tensor} {}

  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr index_type getDimSize(index_type dim) const
  {
    return m_tensor.getDimSize(dim);
  }

  template <typename TILE_TYPE>
  RAJA_INLINE RAJA_HOST_DEVICE result_type eval(TILE_TYPE const &tile) const
  {
    return m_tensor.eval(tile).scale(-1);
  }

  RAJA_INLINE
  RAJA_HOST_DEVICE
  void print_ast() const
  {
    printf("Negate(");
    m_tensor.print_ast();
    printf(")");
  }

private:
  rhs_type m_tensor;
};


}  // namespace ET

}  // namespace expt
}  // namespace internal

}  // namespace RAJA


#endif
