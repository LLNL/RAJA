/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining layout operations for forallN templates.
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
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_util_static_layout_HPP
#define RAJA_util_static_layout_HPP

#include <iostream>
#include <limits>
#include "RAJA/config.hpp"
#include "RAJA/index/IndexValue.hpp"
#include "RAJA/internal/LegacyCompatibility.hpp"
#include "RAJA/util/Operators.hpp"
#include "RAJA/util/Permutations.hpp"

namespace RAJA
{

namespace detail
{


template <typename Range, typename Sizes, typename Strides>
struct StaticLayoutBase_impl;


template <camp::idx_t... RangeInts,
          RAJA::Index_type... Sizes,
          RAJA::Index_type... Strides>
struct StaticLayoutBase_impl<camp::idx_seq<RangeInts...>,
                             camp::idx_seq<Sizes...>,
                             camp::idx_seq<Strides...>> {

  using sizes = camp::idx_seq<Sizes...>;
  using strides = camp::idx_seq<Strides...>;

  /*!
   * Default constructor.
   */
  RAJA_INLINE RAJA_HOST_DEVICE constexpr StaticLayoutBase_impl() {}

  RAJA_INLINE static void print()
  {
    VarOps::ignore_args(printf("SL: arg%d: size=%d, stride=%d\n",
                               (int)RangeInts,
                               (int)Sizes,
                               (int)Strides)...);
  }


  /*!
   * Computes a linear space index from specified indices.
   * This is formed by the dot product of the indices and the layout strides.
   *
   * @param indices  Indices in the n-dimensional space of this layout
   * @return Linear space index.
   */
  template <typename... Indices>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr RAJA::Index_type operator()(
      Indices... indices) const
  {
    // dot product of strides and indices
    return VarOps::sum<RAJA::Index_type>((indices * Strides)...);
  }


  template <typename... Indices>
  static RAJA_INLINE RAJA_HOST_DEVICE constexpr RAJA::Index_type s_oper(
      Indices... indices)
  {
    // dot product of strides and indices
    return VarOps::sum<RAJA::Index_type>((indices * Strides)...);
  }


  /*!
   * Computes a total size of the layout's space.
   * This is the produce of each dimensions size.
   *
   * @return Total size spanned by indices
   */
  RAJA_INLINE RAJA_HOST_DEVICE constexpr RAJA::Index_type size() const
  {
    // Multiply together all of the sizes,
    // replacing 1 for any zero-sized dimensions
    return VarOps::foldl(RAJA::operators::multiplies<RAJA::Index_type>(),
                         (Sizes == 0 ? 1 : Sizes)...);
  }


  static constexpr RAJA::Index_type s_size =
      VarOps::foldl(RAJA::operators::multiplies<RAJA::Index_type>(),
                    (Sizes == 0 ? 1 : Sizes)...);
};

template <camp::idx_t N, camp::idx_t Idx, camp::idx_t... Sizes>
struct StrideCalculatorIdx {
  static_assert(N == sizeof...(Sizes), "");

  using sizes_seq = camp::idx_seq<Sizes...>;
  static constexpr camp::idx_t size = camp::seq_at<Idx, sizes_seq>::value;
  static constexpr camp::idx_t size_last =
      StrideCalculatorIdx<N, Idx + 1, Sizes...>::size;
  static constexpr camp::idx_t value =
      (size_last > 0 ? size_last : 1)
      * StrideCalculatorIdx<N, Idx + 1, Sizes...>::value;
  static constexpr camp::idx_t stride = size > 0 ? value : 0;
};

template <camp::idx_t N, camp::idx_t... Sizes>
struct StrideCalculatorIdx<N, N, Sizes...> {
  static_assert(N == sizeof...(Sizes), "");

  static constexpr camp::idx_t size = 1;
  static constexpr camp::idx_t value = 1;
  static constexpr camp::idx_t stride = size > 0 ? value : 0;
};

template <typename Range, typename Sizes>
struct StrideCalculator;

template <camp::idx_t... RangeInts, camp::idx_t... Sizes>
struct StrideCalculator<camp::idx_seq<RangeInts...>, camp::idx_seq<Sizes...>> {
  static_assert(sizeof...(Sizes) == sizeof...(RangeInts), "");

  using sizes = camp::idx_seq<Sizes...>;
  static constexpr camp::idx_t N = sizeof...(Sizes);
  using strides =
      camp::idx_seq<StrideCalculatorIdx<N, RangeInts, Sizes...>::stride...>;
};


template <typename Layout, typename DimTypeList>
struct TypedStaticLayoutImpl;

template <typename Layout, typename... DimTypes>
struct TypedStaticLayoutImpl<Layout, camp::list<DimTypes...>> {
  /*!
   * Computes a linear space index from specified indices.
   * This is formed by the dot product of the indices and the layout strides.
   *
   * @param indices  Indices in the n-dimensional space of this layout
   * @return Linear space index.
   */
  static RAJA_INLINE RAJA_HOST_DEVICE constexpr RAJA::Index_type s_oper(
      DimTypes... indices)
  {
    return Layout::s_oper(convertIndex<Index_type>(indices)...);
  }


  static constexpr RAJA::Index_type s_size = Layout::s_size;

  RAJA_INLINE
  static void print() { Layout::print(); }
};


}  // namespace detail


template <camp::idx_t... Sizes>
using StaticLayout = detail::
    StaticLayoutBase_impl<camp::make_idx_seq_t<sizeof...(Sizes)>,
                          camp::idx_seq<Sizes...>,
                          typename detail::
                              StrideCalculator<camp::make_idx_seq_t<sizeof...(
                                                   Sizes)>,
                                               camp::idx_seq<Sizes...>>::
                                  strides>;


template <typename TypeList, camp::idx_t... Sizes>
using TypedStaticLayout =
    detail::TypedStaticLayoutImpl<StaticLayout<Sizes...>, TypeList>;


}  // namespace RAJA

#endif
