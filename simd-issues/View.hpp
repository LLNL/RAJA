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

//#include "RAJA/config.hpp"
//#include "RAJA/pattern/atomic.hpp"
#include "Layout.hpp"


namespace RAJA
{

template <typename ValueType,
          typename LayoutType,
          typename PointerType = ValueType *>
struct View {
  using value_type = ValueType;
  using pointer_type = PointerType;
  using layout_type = LayoutType;
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

  RAJA_INLINE void set_data(pointer_type data_ptr) { data = data_ptr; }

  // making this specifically typed would require unpacking the layout,
  // this is easier to maintain
  template <typename... Args>
  RAJA_HOST_DEVICE RAJA_INLINE value_type &operator()(Args... args) const
  {
    return data[(int)convertIndex<Index_type>(layout(args...))];
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


}  // namespace RAJA

#endif
