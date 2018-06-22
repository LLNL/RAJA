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

#ifndef RAJA_VIEW_HPP
#define RAJA_VIEW_HPP

#include <type_traits>

#include "RAJA/config.hpp"
#include "RAJA/pattern/atomic.hpp"
#include "RAJA/util/Layout.hpp"

#if defined(RAJA_ENABLE_CHAI)
#include "chai/ManagedArray.hpp"
#endif

namespace RAJA
{

template <typename ValueType,
          typename LayoutType,
          typename PointerType = ValueType *>
struct View {
  using value_type = ValueType;
  using pointer_type = PointerType;
  using layout_type = LayoutType;
  using nc_value_type = typename std::remove_const<value_type>::type;
  using nc_pointer_type = typename std::add_pointer<
                              typename std::remove_const<
                                  typename std::remove_pointer<pointer_type>::type
                              >::type
                          >::type;
  using NonConstView = View<nc_value_type, layout_type, nc_pointer_type>;

  layout_type const layout;
  pointer_type data;

  template <typename... Args>
  RAJA_INLINE constexpr View(pointer_type data_ptr, Args... dim_sizes)
      : layout(dim_sizes...), data(data_ptr)
  {
  }

  RAJA_INLINE constexpr View(pointer_type data_ptr, layout_type &&layout)
      : layout(layout), data(data_ptr)
  {
  }

  //We found the compiler-generated copy constructor does not actually copy-construct
  //the object on the device in certain nvcc versions. 
  //By explicitly defining the copy constructor we are able ensure proper behavior.
  //Git-hub pull request link https://github.com/LLNL/RAJA/pull/477
  RAJA_INLINE RAJA_HOST_DEVICE constexpr View(View const &V)
      : layout(V.layout), data(V.data)
  {
  }

  template <bool IsConstView = std::is_const<value_type>::value>
  RAJA_INLINE constexpr View(
          typename std::enable_if<IsConstView, NonConstView>::type const &rhs)
      : layout(rhs.layout), data(rhs.data)
  {
  }

  RAJA_INLINE void set_data(pointer_type data_ptr) { data = data_ptr; }

  // making this specifically typed would require unpacking the layout,
  // this is easier to maintain
  template <typename... Args>
  RAJA_HOST_DEVICE RAJA_INLINE value_type &operator()(Args... args) const
  {
    auto idx = convertIndex<Index_type>(layout(args...));
    auto &value = data[idx];
    return value;
  }

  template<typename IndexType, typename VecType>
  RAJA_INLINE
  typename std::conditional<layout_type::stride1_dim == -1, VecType, typename VecType::strided_t>::type
  &operator()(vec::VectorIndex<IndexType, VecType> i) const
  {
    auto &value = data[i.value];
    return reinterpret_cast<VecType &>(value);
  }

  template<typename IndexType, typename T>
  RAJA_INLINE value_type& operator()(vec::VectorIndex<IndexType, vec::Vector<T,1,1>> i) const
  {
    return data[i.value];
  }

};

template <typename ValueType,
          typename PointerType,
          typename LayoutType,
          typename... IndexTypes>
struct TypedViewBase {
  using Base = View<ValueType, LayoutType, PointerType>;

  Base base_;

  template <typename... Args>
  RAJA_INLINE constexpr TypedViewBase(PointerType data_ptr, Args... dim_sizes)
      : base_(data_ptr, dim_sizes...)
  {
  }

  template <typename CLayoutType>
  RAJA_INLINE constexpr TypedViewBase(PointerType data_ptr,
                                      CLayoutType &&layout)
      : base_(data_ptr, std::forward<CLayoutType>(layout))
  {
  }

  RAJA_INLINE void set_data(PointerType data_ptr) { base_.set_data(data_ptr); }

  RAJA_HOST_DEVICE RAJA_INLINE ValueType &operator()(IndexTypes... args) const
  {
    return base_.operator()(convertIndex<Index_type>(args)...);
  }
};

template <typename ValueType, typename LayoutType, typename... IndexTypes>
using TypedView =
    TypedViewBase<ValueType, ValueType *, LayoutType, IndexTypes...>;

#if defined(RAJA_ENABLE_CHAI)

template <typename ValueType, typename LayoutType>
using ManagedArrayView =
    View<ValueType, LayoutType, chai::ManagedArray<ValueType>>;


template <typename ValueType, typename LayoutType, typename... IndexTypes>
using TypedManagedArrayView = TypedViewBase<ValueType,
                                            chai::ManagedArray<ValueType>,
                                            LayoutType,
                                            IndexTypes...>;

#endif


template <typename ViewType, typename AtomicPolicy = RAJA::atomic::auto_atomic>
struct AtomicViewWrapper {
  using base_type = ViewType;
  using pointer_type = typename base_type::pointer_type;
  using value_type = typename base_type::value_type;
  using atomic_type = RAJA::atomic::AtomicRef<value_type, AtomicPolicy>;

  base_type base_;

  RAJA_INLINE
  constexpr explicit AtomicViewWrapper(ViewType view) : base_(view) {}

  RAJA_INLINE void set_data(pointer_type data_ptr) { base_.set_data(data_ptr); }

  template <typename... ARGS>
  RAJA_HOST_DEVICE RAJA_INLINE atomic_type operator()(ARGS &&... args) const
  {
    return atomic_type(&base_.operator()(std::forward<ARGS>(args)...));
  }
};


/*
 * Specialized AtomicViewWrapper for seq_atomic that acts as pass-thru
 * for performance
 */
template <typename ViewType>
struct AtomicViewWrapper<ViewType, RAJA::atomic::seq_atomic> {
  using base_type = ViewType;
  using pointer_type = typename base_type::pointer_type;
  using value_type = typename base_type::value_type;
  using atomic_type =
      RAJA::atomic::AtomicRef<value_type, RAJA::atomic::seq_atomic>;

  base_type base_;

  RAJA_INLINE
  constexpr explicit AtomicViewWrapper(ViewType const &view) : base_{view} {}

  RAJA_INLINE void set_data(pointer_type data_ptr) { base_.set_data(data_ptr); }

  template <typename... ARGS>
  RAJA_HOST_DEVICE RAJA_INLINE value_type &operator()(ARGS &&... args) const
  {
    return base_.operator()(std::forward<ARGS>(args)...);
  }
};


template <typename AtomicPolicy, typename ViewType>
RAJA_INLINE AtomicViewWrapper<ViewType, AtomicPolicy> make_atomic_view(
    ViewType const &view)
{

  return RAJA::AtomicViewWrapper<ViewType, AtomicPolicy>(view);
}

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
