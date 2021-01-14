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

#ifndef RAJA_pattern_tensor_TensorBlock_HPP
#define RAJA_pattern_tensor_TensorBlock_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#include "camp/camp.hpp"
#include "RAJA/pattern/tensor/TensorLayout.hpp"
#include "RAJA/pattern/tensor/internal/ET/TensorRef.hpp"
#include "RAJA/util/StaticLayout.hpp"

namespace RAJA
{


  template<typename T, typename IDX_TYPES, typename LAYOUT>
  class TensorStackStorage : public RAJA::internal::TypedViewBase<T, T*, LAYOUT, IDX_TYPES>
  {
    public:
      using self_type = TensorStackStorage<T, IDX_TYPES, LAYOUT>;
      using base_type = RAJA::internal::TypedViewBase<T, T*, LAYOUT, IDX_TYPES>;
      using element_type = T;
      using layout_type = LAYOUT;

      static constexpr camp::idx_t s_num_elem = LAYOUT::s_size();

      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      TensorStackStorage() noexcept :
        base_type(&m_data[0], layout_type{})
      {
      }


//      RAJA_HOST_DEVICE
//      RAJA_INLINE
//      constexpr
//      self_type createTemporary() const noexcept{
//        return self_type{};
//      }

    private:
      T m_data[s_num_elem];
  };


  template<typename REGISTER_POLICY,
           typename ELEMENT_TYPE,
           typename LAYOUT,
           typename STORAGE,
           typename ITER_STRATEGY>
  class TensorBlock;


  template<typename REGISTER_POLICY, typename T, typename LAYOUT, typename STORAGE, typename ITER_STRATEGY>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  TensorBlock<REGISTER_POLICY, T, LAYOUT, STORAGE, ITER_STRATEGY>
  operator+(T x, TensorBlock<REGISTER_POLICY, T, LAYOUT, STORAGE, ITER_STRATEGY> const &y){
    using block_t = TensorBlock<REGISTER_POLICY, T, LAYOUT, STORAGE, ITER_STRATEGY>;
    return block_t(x).add(y);
  }

  template<typename REGISTER_POLICY, typename T, typename LAYOUT, typename STORAGE, typename ITER_STRATEGY>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  TensorBlock<REGISTER_POLICY, T, LAYOUT, STORAGE, ITER_STRATEGY>
  operator-(T x, TensorBlock<REGISTER_POLICY, T, LAYOUT, STORAGE, ITER_STRATEGY> const &y){
    using block_t = TensorBlock<REGISTER_POLICY, T, LAYOUT, STORAGE, ITER_STRATEGY>;
    return block_t(x).subtract(y);
  }

  template<typename REGISTER_POLICY, typename T, typename LAYOUT, typename STORAGE, typename ITER_STRATEGY>
  TensorBlock<REGISTER_POLICY, T, LAYOUT, STORAGE, ITER_STRATEGY>
  operator*(T x, TensorBlock<REGISTER_POLICY, T, LAYOUT, STORAGE, ITER_STRATEGY> const &y){
    using block_t = TensorBlock<REGISTER_POLICY, T, LAYOUT, STORAGE, ITER_STRATEGY>;
    return block_t(x).multiply(y);
  }

  template<typename REGISTER_POLICY, typename T, typename LAYOUT, typename STORAGE, typename ITER_STRATEGY>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  TensorBlock<REGISTER_POLICY, T, LAYOUT, STORAGE, ITER_STRATEGY>
  operator/(T x, TensorBlock<REGISTER_POLICY, T, LAYOUT, STORAGE, ITER_STRATEGY> const &y){
    using block_t = TensorBlock<REGISTER_POLICY, T, LAYOUT, STORAGE, ITER_STRATEGY>;
    return block_t(x).divide(y);
  }



  template<typename REGISTER_POLICY,
           typename ELEMENT_TYPE,
           camp::idx_t ... LAYOUT,
           camp::idx_t ... DIM_SIZES,
           typename STORAGE,
           typename ITER_STRATEGY>
  class TensorBlock<REGISTER_POLICY,
                    ELEMENT_TYPE,
                    RAJA::StaticLayout<camp::idx_seq<LAYOUT...>, DIM_SIZES...>,
                    STORAGE,
                    ITER_STRATEGY
                    >
  {
    public:

    private:
      STORAGE m_storage;

    public:

  };

}  // namespace RAJA



#endif
