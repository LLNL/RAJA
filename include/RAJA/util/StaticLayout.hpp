/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining Layout, a N-dimensional index calculator
 *          with compile-time defined sizes and permutation
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_util_static_layout_HPP
#define RAJA_util_static_layout_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <limits>

#include "RAJA/index/IndexValue.hpp"

#include "RAJA/internal/foldl.hpp"

#include "RAJA/util/Operators.hpp"
#include "RAJA/util/Permutations.hpp"

namespace RAJA
{

namespace detail
{


template <typename IndexType, typename Range, typename Sizes, typename Strides>
struct StaticLayoutBase_impl;


template <typename IndexType,
          camp::idx_t... RangeInts,
          camp::idx_t... Sizes,
          camp::idx_t... Strides>
struct StaticLayoutBase_impl<IndexType,
                             camp::idx_seq<RangeInts...>,
                             camp::idx_seq<Sizes...>,
                             camp::idx_seq<Strides...>> {

  using IndexLinear = IndexType;
  using sizes = camp::int_seq<IndexType, ((IndexType)Sizes)...>;
  using strides = camp::int_seq<IndexType, ((IndexType)Strides)...>;

  static constexpr size_t n_dims = sizeof...(Sizes);

  /*!
   * Default constructor.
   */
  RAJA_INLINE RAJA_HOST_DEVICE constexpr StaticLayoutBase_impl() {}

  RAJA_INLINE static void print()
  {
    camp::sink(printf("StaticLayout: arg%d: size=%d, stride=%d\n",
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
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexLinear operator()(
      Indices... indices) const
  {
    // dot product of strides and indices
    return VarOps::sum<IndexLinear>((indices * Strides)...);
  }


  template <typename... Indices>
  static RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexLinear s_oper(Indices... indices)
  {
    // dot product of strides and indices
    return VarOps::sum<IndexLinear>((indices * Strides)...);
  }


  /*!
   * Computes a total size of the layout's space.
   * This is the produce of each dimensions size.
   *
   * @return Total size spanned by indices
   */
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexLinear size() const
  {

    return s_size;
  }


  static constexpr IndexLinear s_size =
      foldl(RAJA::operators::multiplies<IndexLinear>(),
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
      (size_last > 0 ? size_last : 1) *
      StrideCalculatorIdx<N, Idx + 1, Sizes...>::value;
  static constexpr camp::idx_t stride = size > 0 ? value : 0;
};

template <camp::idx_t N, camp::idx_t... Sizes>
struct StrideCalculatorIdx<N, N, Sizes...> {
  static_assert(N == sizeof...(Sizes), "");

  static constexpr camp::idx_t size = 1;
  static constexpr camp::idx_t value = 1;
  static constexpr camp::idx_t stride = size > 0 ? value : 0;
};

template <typename Range, typename Perm, typename Sizes>
struct StrideCalculator;

template <camp::idx_t ... Range, camp::idx_t... Perm, camp::idx_t... Sizes>
struct StrideCalculator<camp::idx_seq<Range...>, camp::idx_seq<Perm...>, camp::idx_seq<Sizes...>> {
  static_assert(sizeof...(Sizes) == sizeof...(Perm), "");

  using sizes = camp::idx_seq<Sizes...>;
  static constexpr camp::idx_t N = sizeof...(Sizes);
  using range = camp::idx_seq<Range...>;
  using perm = camp::idx_seq<Perm...>;
  using inv_perm = invert_permutation<perm>;
  using strides_unperm =
      camp::idx_seq<StrideCalculatorIdx<N, Range, camp::seq_at<Perm, sizes>::value...>::stride...>;
  
  using strides = camp::idx_seq<camp::seq_at<camp::seq_at<Range, inv_perm>::value, strides_unperm>::value...>;
};


template <typename Layout, typename DimTypeList>
struct TypedStaticLayoutImpl;

template <typename Layout, typename... DimTypes>
struct TypedStaticLayoutImpl<Layout, camp::list<DimTypes...>> {

  using IndexLinear = typename Layout::IndexType;

  static constexpr IndexLinear n_dims = sizeof...(DimTypes);

  /*!
   * Computes a linear space index from specified indices.
   * This is formed by the dot product of the indices and the layout strides.
   *
   * @param indices  Indices in the n-dimensional space of this layout
   * @return Linear space index.
   */
  static RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexLinear s_oper(
      DimTypes... indices)
  {
    return Layout::s_oper(stripIndexType(indices)...);
  }


  static constexpr IndexLinear s_size = Layout::s_size;

  RAJA_INLINE
  static void print() { Layout::print(); }
};


}  // namespace detail


template <typename Perm, camp::idx_t... Sizes>
using StaticLayout = detail::StaticLayoutBase_impl<
    camp::idx_t,
    camp::make_idx_seq_t<sizeof...(Sizes)>,
    camp::idx_seq<Sizes...>,
    typename detail::StrideCalculator<camp::make_idx_seq_t<sizeof...(Sizes)>,
                                      Perm,
                                      camp::idx_seq<Sizes...>>::strides>;






template <typename Perm, typename TypeList, camp::idx_t... Sizes>
using TypedStaticLayout =
    detail::TypedStaticLayoutImpl<StaticLayout<Perm, Sizes...>, TypeList>;


}  // namespace RAJA

#endif
