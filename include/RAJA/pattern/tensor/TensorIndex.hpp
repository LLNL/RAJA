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
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_tensor_TensorIndex_HPP
#define RAJA_pattern_tensor_TensorIndex_HPP

#include "RAJA/config.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/index/IndexValue.hpp"


namespace RAJA
{
namespace expt
{
  template<typename IDX, typename TENSOR_TYPE, camp::idx_t DIM>
  class TensorIndex {
    public:
      using self_type = TensorIndex<IDX, TENSOR_TYPE, DIM>;
      using value_type = strip_index_type_t<IDX>;
      using index_type = IDX;
      using tensor_type = TENSOR_TYPE;

      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      constexpr
      self_type all(){
        return self_type(index_type(-1), value_type(-1));
      }

      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      constexpr
      self_type range(index_type begin, index_type end){
        return self_type(begin, value_type(stripIndexType(end-begin)));
      }


      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      TensorIndex() : m_index(index_type(0)), m_length(0) {}


      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      TensorIndex(RAJA::TypedRangeSegment<IDX> const &seg) :
      m_index(*seg.begin()), m_length(seg.size())
      {}


      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      TensorIndex(index_type value, value_type length) : m_index(value), m_length(length) {}

      template<typename T, camp::idx_t D>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      TensorIndex(TensorIndex<IDX, T, D> const &c) : m_index(*c), m_length(c.size()) {}


      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      index_type const &operator*() const {
        return m_index;
      }

      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      index_type begin() const {
        return m_index;
      }

      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      value_type size() const {
        return m_length;
      }

      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      value_type dim() const {
        return DIM;
      }

    private:
      index_type m_index;
      value_type m_length;
  };


  /*!
   * Index that specifies the starting element index of a Vector
   */
  template<typename IDX, typename VECTOR_TYPE>
  using VectorIndex =  TensorIndex<IDX, VECTOR_TYPE, 0>;

  /*!
   * Index that specifies the starting Row index of a matrix
   */
  template<typename IDX, typename MATRIX_TYPE>
  using RowIndex =  TensorIndex<IDX, MATRIX_TYPE, 0>;

  /*!
   * Index that specifies the starting Column index of a matrix
   */
  template<typename IDX, typename MATRIX_TYPE>
  using ColIndex =  TensorIndex<IDX, MATRIX_TYPE, 1>;


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

} // namespace expt
}  // namespace RAJA

#include "RAJA/pattern/tensor/internal/TensorIndexTraits.hpp"

#endif
