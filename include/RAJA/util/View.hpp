/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining a multi-dimensional view class.
 *
 ******************************************************************************
 */

#ifndef RAJA_VIEW_HPP
#define RAJA_VIEW_HPP

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
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

#include "RAJA/config.hpp"
#include "RAJA/util/Layout.hpp"

#if defined(RAJA_ENABLE_CHAI)
#include "chai/ManagedArray.hpp"
#endif

namespace RAJA
{

template <typename DataType,
          typename LayoutT,
          typename DataPointer = DataType *>
struct View {
  LayoutT const layout;
  DataPointer data;

  template <typename... Args>
  RAJA_INLINE constexpr View(DataPointer data_ptr, Args... dim_sizes)
      : layout(dim_sizes...), data(data_ptr)
  {
  }

  RAJA_INLINE constexpr View(DataPointer data_ptr, LayoutT &&layout)
      : layout(layout), data(data_ptr)
  {
  }

  RAJA_INLINE void set_data(DataPointer data_ptr) { data = data_ptr; }

  // making this specifically typed would require unpacking the layout,
  // this is easier to maintain
  template <typename... Args>
  RAJA_HOST_DEVICE RAJA_INLINE DataType &operator()(Args... args) const
  {
    return data[convertIndex<Index_type>(layout(args...))];
  }
};

template <typename DataType,
          typename DataPointer,
          typename LayoutT,
          typename... IndexTypes>
struct TypedViewBase {
  using Base = View<DataType, LayoutT, DataPointer>;

  Base base_;

  template <typename... Args>
  RAJA_INLINE constexpr TypedViewBase(DataPointer data_ptr, Args... dim_sizes)
      : base_(data_ptr, dim_sizes...)
  {
  }

  RAJA_INLINE constexpr TypedViewBase(DataPointer data_ptr, LayoutT &&layout)
      : base_(data_ptr, layout)
  {
  }

  RAJA_INLINE void set_data(DataPointer data_ptr) { base_.set_data(data_ptr); }

  RAJA_HOST_DEVICE RAJA_INLINE DataType &operator()(IndexTypes... args) const
  {
    return base_.operator()(convertIndex<Index_type>(args)...);
  }
};

template <typename DataType, typename LayoutT, typename... IndexTypes>
using TypedView = TypedViewBase<DataType, DataType *, LayoutT, IndexTypes...>;

#if defined(RAJA_ENABLE_CHAI)

template <typename DataType, typename LayoutT>
using ManagedArrayView = View<DataType, LayoutT, chai::ManagedArray<DataType>>;


template <typename DataType, typename LayoutT, typename... IndexTypes>
using TypedManagedArrayView = TypedViewBase<DataType,
                                            chai::ManagedArray<DataType>,
                                            LayoutT,
                                            IndexTypes...>;

#endif


}  // namespace RAJA

#endif
