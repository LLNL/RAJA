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
// Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_VIEW_HPP
#define RAJA_VIEW_HPP

#include "RAJA/config.hpp"
#include "RAJA/util/Layout.hpp"
#include "RAJA/pattern/atomic.hpp"

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
  layout_type const layout;
  pointer_type data;

  using junk = typename LayoutType::IndexRange;

  template <typename... Args>
  RAJA_INLINE constexpr View(pointer_type data_ptr, Args... dim_sizes)
      : layout(dim_sizes...), data(data_ptr)
  {
  }

  RAJA_INLINE constexpr View(pointer_type data_ptr, layout_type &&layout)
      : layout(layout), data(data_ptr)
  {
  }

  template< typename InputLayoutType >
  RAJA_INLINE constexpr View( View<ValueType,InputLayoutType,PointerType> const & input )
      : layout(input.layout), data(input.data)
  {
  }


  RAJA_INLINE void set_data(pointer_type data_ptr) { data = data_ptr; }

  // making this specifically typed would require unpacking the layout,
  // this is easier to maintain
  template <typename... Args>
  RAJA_HOST_DEVICE RAJA_INLINE value_type &operator()(Args... args) const
  {
    return data[convertIndex<Index_type>(layout(args...))];
  }


  template< size_t NDIM = layout_type::n_dims >
  RAJA_HOST_DEVICE RAJA_INLINE
  typename std::enable_if<  NDIM != 1,
                            View<ValueType, typename layout_type::sliced_layout, PointerType> >::type
  operator[]( typename layout_type::IndexLinear index ) const
  {
    return View<ValueType, typename layout_type::sliced_layout, PointerType>( data+index*layout.strides[0], layout[index] );
  }

  template< size_t NDIM = layout_type::n_dims >
  RAJA_HOST_DEVICE RAJA_INLINE
  typename std::enable_if< NDIM == 1, ValueType & >::type
  operator[]( typename layout_type::IndexLinear index ) const
  {
    return data[index];
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

  RAJA_INLINE constexpr TypedViewBase(PointerType data_ptr, LayoutType &&layout)
      : base_(data_ptr, layout)
  {
  }

  RAJA_INLINE void set_data(PointerType data_ptr) { base_.set_data(data_ptr); }

  RAJA_HOST_DEVICE RAJA_INLINE ValueType &operator()(IndexTypes... args) const
  {
    return base_.operator()(convertIndex<Index_type>(args)...);
  }
};

template <typename ValueType, typename LayoutType, typename... IndexTypes>
using TypedView = TypedViewBase<ValueType, ValueType *, LayoutType, IndexTypes...>;

#if defined(RAJA_ENABLE_CHAI)

template <typename ValueType, typename LayoutType>
using ManagedArrayView = View<ValueType, LayoutType, chai::ManagedArray<ValueType>>;


template <typename ValueType, typename LayoutType, typename... IndexTypes>
using TypedManagedArrayView = TypedViewBase<ValueType,
                                            chai::ManagedArray<ValueType>,
                                            LayoutType,
                                            IndexTypes...>;

#endif


template <typename ViewType,
          typename AtomicPolicy = RAJA::atomic::auto_atomic>
struct AtomicViewWrapper {
  using base_type = ViewType;
  using pointer_type = typename base_type::pointer_type;
  using value_type = typename base_type::value_type;
  using atomic_type = RAJA::atomic::AtomicRef<value_type, AtomicPolicy>;

  base_type base_;

  RAJA_INLINE
  constexpr
  explicit
  AtomicViewWrapper(ViewType view)
      : base_(view)
  {
  }

  RAJA_INLINE void set_data(pointer_type data_ptr) { base_.set_data(data_ptr); }

  template<typename ... ARGS>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  atomic_type operator()(ARGS &&... args) const
  {
    return atomic_type(&base_.operator()(std::forward<ARGS>(args)...));
  }

};


template<typename AtomicPolicy, typename ViewType>
RAJA_INLINE
AtomicViewWrapper<ViewType, AtomicPolicy>
make_atomic_view(ViewType view){
  return RAJA::AtomicViewWrapper<ViewType, AtomicPolicy>(view);
}


}  // namespace RAJA

#endif
