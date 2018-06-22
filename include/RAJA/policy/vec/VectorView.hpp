/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining a multi-dimensional view class.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_vec_vectorview_HPP
#define RAJA_policy_vec_vectorview_HPP

#include <type_traits>

#include "RAJA/config.hpp"
#include "RAJA/util/View.hpp"

namespace RAJA
{

namespace internal {

template<typename VectorType>
struct MakeVectorHelper;

template<typename T, size_t N, size_t U>
struct MakeVectorHelper<RAJA::vec::Vector<T, N, U>&> {
    using vector_type = RAJA::vec::Vector<T, N, U>;

    RAJA_INLINE
    static
    vector_type &make_vector(T &ref, size_t ) {
      reinterpret_cast<vector_type &>(ref);
    }
};

template<typename T, size_t N, size_t U>
struct MakeVectorHelper<RAJA::vec::StridedVector<T, N, U>> {
    using vector_type = RAJA::vec::StridedVector<T, N, U>;

    RAJA_INLINE
    static
    vector_type make_vector(T &ref, size_t stride) {
      return vector_type {&ref, stride};
    }
};

template<typename T>
struct StripVectorIndexHelper{

    using value_type = T;

    RAJA_INLINE
    static
    constexpr
    value_type strip(T const &v){
      return v;
    }
};

template<typename IdxType, typename VecType>
struct StripVectorIndexHelper<RAJA::vec::VectorIndex<IdxType, VecType>>{

    using vec_idx_t = RAJA::vec::VectorIndex<IdxType, VecType>;

    using value_type = IdxType;

    RAJA_INLINE
    static
    constexpr
    value_type strip(vec_idx_t const &idx) {
      return idx.value;
    }
};


template<typename T>
auto strip_vector_index(T const &idx) ->
  typename internal::StripVectorIndexHelper<T>::value_type
{
  return internal::StripVectorIndexHelper<T>::strip(idx);
}



} // namespace internal

template <typename ViewType, typename VectorType, RAJA::idx_t VectorDim>
struct VectorViewWrapper{
  using base_type = ViewType;
  using pointer_type = typename base_type::pointer_type;
  using value_type = typename base_type::value_type;

  // Choose a vector or strided vector type based on what the stride1 dim is
  static constexpr RAJA::idx_t stride1_dim = ViewType::layout_type::stride1_dim;

  using vector_type = typename std::conditional<stride1_dim == VectorDim, VectorType&, typename VectorType::strided_type>::type;
  using scalar_type = typename VectorType::scalar_type;

  base_type base_;

  RAJA_INLINE
  constexpr explicit VectorViewWrapper(ViewType const &view) : base_{view} {}

  RAJA_INLINE void set_data(pointer_type data_ptr) { base_.set_data(data_ptr); }

  template <typename... ARGS>
  RAJA_HOST_DEVICE RAJA_INLINE vector_type operator()(ARGS &&... args) const
  {
    auto &val = base_.operator()(internal::strip_vector_index(args)...);
    return internal::MakeVectorHelper<vector_type>::make_vector(val, base_.layout.strides[VectorDim]);
  }


};

template < typename VectorType, RAJA::idx_t VectorDim, typename ViewType>
RAJA_INLINE VectorViewWrapper<ViewType, VectorType, VectorDim> make_vector_view(
    ViewType const &view)
{

  return RAJA::VectorViewWrapper<ViewType, VectorType, VectorDim>(view);
}



}  // namespace RAJA

#endif
