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
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_VIEW_HPP
#define RAJA_VIEW_HPP

#include <type_traits>

#include "RAJA/config.hpp"

#include "RAJA/pattern/atomic.hpp"

#include "RAJA/util/Layout.hpp"
#include "RAJA/util/OffsetLayout.hpp"

namespace RAJA
{

//Helpers to convert
//layouts -> OffsetLayouts
//Typedlayouts -> TypedOffsetLayouts
template<typename layout>
struct add_offset
{
  using type = RAJA::OffsetLayout<layout::n_dims>;
};

template<typename IdxLin, typename...DimTypes>
struct add_offset<RAJA::TypedLayout<IdxLin,camp::tuple<DimTypes...>>>
{
  using type = RAJA::TypedOffsetLayout<IdxLin,camp::tuple<DimTypes...>>;
};

template <typename ValueType,
          typename LayoutType,
          typename PointerType = ValueType *>
struct View {
  using value_type = ValueType;
  using pointer_type = PointerType;
  using layout_type = LayoutType;
  using nc_value_type = typename std::remove_const<value_type>::type;
  using nc_pointer_type = typename std::add_pointer<typename std::remove_const<
      typename std::remove_pointer<pointer_type>::type>::type>::type;
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

  // We found the compiler-generated copy constructor does not actually
  // copy-construct the object on the device in certain nvcc versions. By
  // explicitly defining the copy constructor we are able ensure proper
  // behavior. Git-hub pull request link https://github.com/LLNL/RAJA/pull/477
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

  template <size_t n_dims=layout_type::n_dims, typename IdxLin = Index_type>
  RAJA_INLINE RAJA::View<ValueType, typename add_offset<layout_type>::type>
  shift(const std::array<IdxLin, n_dims>& shift)
  {
    static_assert(n_dims==layout_type::n_dims, "Dimension mismatch in view shift");

    typename add_offset<layout_type>::type shift_layout(layout);
    shift_layout.shift(shift);

    return RAJA::View<ValueType, typename add_offset<layout_type>::type>(data, shift_layout);
  }

  // making this specifically typed would require unpacking the layout,
  // this is easier to maintain
  template <typename... Args>
  RAJA_HOST_DEVICE RAJA_INLINE value_type &operator()(Args... args) const
  {
    auto idx = stripIndexType(layout(args...));
    return data[idx];
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

  template <size_t n_dims=Base::layout_type::n_dims, typename IdxLin = Index_type>
  RAJA_INLINE RAJA::TypedViewBase<ValueType, ValueType *, typename add_offset<LayoutType>::type, IndexTypes...>
  shift(const std::array<IdxLin, n_dims>& shift)
  {
    static_assert(n_dims==Base::layout_type::n_dims, "Dimension mismatch in view shift");

    typename add_offset<LayoutType>::type shift_layout(base_.layout);
    shift_layout.shift(shift);

    return RAJA::TypedViewBase<ValueType, ValueType *, typename add_offset<LayoutType>::type, IndexTypes...>(base_.data, shift_layout);
  }

  RAJA_HOST_DEVICE RAJA_INLINE ValueType &operator()(IndexTypes... args) const
  {
    return base_.operator()(stripIndexType(args)...);
  }
};

template <typename ValueType, typename LayoutType, typename... IndexTypes>
using TypedView =
    TypedViewBase<ValueType, ValueType *, LayoutType, IndexTypes...>;

template <typename ViewType, typename AtomicPolicy = RAJA::auto_atomic>
struct AtomicViewWrapper {
  using base_type = ViewType;
  using pointer_type = typename base_type::pointer_type;
  using value_type = typename base_type::value_type;
  using atomic_type = RAJA::AtomicRef<value_type, AtomicPolicy>;

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
struct AtomicViewWrapper<ViewType, RAJA::seq_atomic> {
  using base_type = ViewType;
  using pointer_type = typename base_type::pointer_type;
  using value_type = typename base_type::value_type;
  using atomic_type = RAJA::AtomicRef<value_type, RAJA::seq_atomic>;

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


}  // namespace RAJA

#endif
