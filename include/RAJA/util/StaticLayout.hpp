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
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
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
#include "RAJA/util/Layout.hpp"


namespace RAJA
{

namespace detail
{

// helper for calculating stride_max_dim using RAJA::max
// operator >= only compares stride, but brings along the index
template <typename IdxLin>
struct StrideMaxDimHelper
{
  IdxLin stride;
  camp::idx_t idx;

  RAJA_HOST_DEVICE constexpr StrideMaxDimHelper(IdxLin stride_, camp::idx_t idx_)
    : stride(stride_), idx(idx_)
  { }

  RAJA_HOST_DEVICE constexpr bool operator>=(StrideMaxDimHelper const& rhs) const
  {
    return stride >= rhs.stride;
  }
};

} // end namespace detail

namespace operators
{

// partially specialize limits to work with StrideMaxDimHelper
template <typename IdxLin>
struct limits<::RAJA::detail::StrideMaxDimHelper<IdxLin>> : limits<IdxLin> { };

} // end namespace operators

namespace detail
{

template <typename IdxLin, IdxLin N, IdxLin Idx, IdxLin... Sizes>
struct StrideCalculatorNoProjIdx {
  static_assert(N == sizeof...(Sizes), "");

  using sizes_seq = camp::int_seq<IdxLin, Sizes...>;
  static constexpr IdxLin size = camp::seq_at<Idx, sizes_seq>::value;
  static constexpr IdxLin size_last =
      StrideCalculatorNoProjIdx<IdxLin, N, Idx + 1, Sizes...>::size;
  static constexpr IdxLin value =
      size_last *
      StrideCalculatorNoProjIdx<IdxLin, N, Idx + 1, Sizes...>::value;
  static constexpr IdxLin stride = value;
};

template <typename IdxLin, IdxLin N, IdxLin... Sizes>
struct StrideCalculatorNoProjIdx<IdxLin, N, N, Sizes...> {
  static_assert(N == sizeof...(Sizes), "");

  static constexpr IdxLin size = 1;
  static constexpr IdxLin value = 1;
  static constexpr IdxLin stride = value;
};

template <typename IdxLin, typename Range, typename Perm, typename Sizes>
struct StrideCalculatorNoProj;

template <typename IdxLin, IdxLin ... Range, camp::idx_t... Perm, IdxLin... Sizes>
struct StrideCalculatorNoProj<IdxLin,
                        camp::int_seq<IdxLin, Range...>,
                        camp::idx_seq<Perm...>,
                        camp::int_seq<IdxLin, Sizes...>> {
  static_assert(sizeof...(Sizes) == sizeof...(Perm), "");

  using sizes = camp::int_seq<IdxLin, Sizes...>;
  static constexpr IdxLin N = sizeof...(Sizes);
  using range = camp::int_seq<IdxLin, Range...>;
  using perm = camp::idx_seq<Perm...>;
  using inv_perm = invert_permutation<perm>;
  using strides_unperm =
      camp::int_seq<IdxLin, StrideCalculatorNoProjIdx<IdxLin, N, Range, camp::seq_at<Perm, sizes>::value...>::stride...>;

  using strides = camp::int_seq<IdxLin, camp::seq_at<camp::seq_at<Range, inv_perm>::value, strides_unperm>::value...>;
};


template <typename IdxLin, IdxLin N, IdxLin Idx, IdxLin... Sizes>
struct StrideCalculatorIdx {
  static_assert(N == sizeof...(Sizes), "");

  using sizes_seq = camp::int_seq<IdxLin, Sizes...>;
  static constexpr IdxLin size = camp::seq_at<Idx, sizes_seq>::value;
  static constexpr IdxLin size_last =
      StrideCalculatorIdx<IdxLin, N, Idx + 1, Sizes...>::size;
  static constexpr IdxLin value =
      (size_last > 0 ? size_last : 1) *
      StrideCalculatorIdx<IdxLin, N, Idx + 1, Sizes...>::value;
  static constexpr IdxLin stride = size > 0 ? value : 0;
};

template <typename IdxLin, IdxLin N, IdxLin... Sizes>
struct StrideCalculatorIdx<IdxLin, N, N, Sizes...> {
  static_assert(N == sizeof...(Sizes), "");

  static constexpr IdxLin size = 1;
  static constexpr IdxLin value = 1;
  static constexpr IdxLin stride = size > 0 ? value : 0;
};

template <typename IdxLin, typename Range, typename Perm, typename Sizes>
struct StrideCalculator;

template <typename IdxLin, IdxLin ... Range, camp::idx_t... Perm, IdxLin... Sizes>
struct StrideCalculator<IdxLin,
                        camp::int_seq<IdxLin, Range...>,
                        camp::idx_seq<Perm...>,
                        camp::int_seq<IdxLin, Sizes...>> {
  static_assert(sizeof...(Sizes) == sizeof...(Perm), "");

  using sizes = camp::int_seq<IdxLin, Sizes...>;
  static constexpr IdxLin N = sizeof...(Sizes);
  using range = camp::int_seq<IdxLin, Range...>;
  using perm = camp::idx_seq<Perm...>;
  using inv_perm = invert_permutation<perm>;
  using strides_unperm =
      camp::int_seq<IdxLin, StrideCalculatorIdx<IdxLin, N, Range, camp::seq_at<Perm, sizes>::value...>::stride...>;

  using strides = camp::int_seq<IdxLin, camp::seq_at<camp::seq_at<Range, inv_perm>::value, strides_unperm>::value...>;
};


template <typename IdxLin, typename Range, typename Sizes, typename Strides>
struct StaticLayoutNoProjBase_impl;

template <typename IdxLin,
          IdxLin... RangeInts,
          IdxLin... Sizes,
          IdxLin... Strides>
struct StaticLayoutNoProjBase_impl<IdxLin,
                             camp::int_seq<IdxLin, RangeInts...>,
                             camp::int_seq<IdxLin, Sizes...>,
                             camp::int_seq<IdxLin, Strides...>>
    : LayoutBaseMarker
{
  using Self = StaticLayoutNoProjBase_impl<IdxLin,
                             camp::int_seq<IdxLin, RangeInts...>,
                             camp::int_seq<IdxLin, Sizes...>,
                             camp::int_seq<IdxLin, Strides...>>;

  using IndexRange = camp::make_idx_seq_t<sizeof...(RangeInts)>;
  using StrippedIndexLinear = IdxLin;
  using IndexLinear = IdxLin;
  using DimTuple = camp::tuple<FirstTemplateParam<IndexLinear, RangeInts>...>;
  using DimArr = std::array<StrippedIndexLinear, sizeof...(RangeInts)>;

  static_assert(std::is_same<camp::idx_seq<RangeInts...>, IndexRange>::value,
      "Range must in order");
  static_assert(sizeof...(RangeInts) == sizeof...(Sizes),
      "RangeInts and Sizes must be the same size");
  static_assert(sizeof...(RangeInts) == sizeof...(Strides),
      "RangeInts and Strides must be the same size");

  using sizes = camp::int_seq<IdxLin, Sizes...>;
  using strides = camp::int_seq<IdxLin, Strides...>;

  static constexpr size_t n_dims = sizeof...(Sizes);
  static constexpr camp::idx_t stride_one_dim =
      RAJA::max<camp::idx_t>(
          (camp::seq_at<RangeInts, strides>::value == 1 ? camp::idx_t{RangeInts} : camp::idx_t{-1})...);
  static constexpr camp::idx_t stride_max_dim =
      RAJA::max<StrideMaxDimHelper<IdxLin>>(
          StrideMaxDimHelper<IdxLin>{ camp::seq_at<RangeInts, strides>::value, camp::idx_t{RangeInts} }...).idx;


  /*!
   * Default constructor.
   */
  RAJA_INLINE RAJA_HOST_DEVICE constexpr StaticLayoutNoProjBase_impl() {}

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
  RAJA_INLINE RAJA_HOST_DEVICE static constexpr IdxLin s_oper(Indices... indices)
  {
    static_assert(n_dims == sizeof...(Indices),
        "Error: wrong number of indices");
    // dot product of strides and indices
    return RAJA::sum<IdxLin>((IdxLin(indices * Strides))...);
  }
  ///
  template <typename... Indices>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin operator()(
      Indices... indices) const
  {
    static_assert(n_dims == sizeof...(Indices),
        "Error: wrong number of indices");
    return s_oper(indices...);
  }

  // Multiply together all of the sizes,
  // replacing 1 for any zero-sized dimensions
  static constexpr IdxLin s_size =
      RAJA::product<IdxLin>((Sizes == IdxLin(0) ? IdxLin(1) : Sizes)...);

  /*!
   * Computes a total size of the layout's space.
   * This is the produce of each dimensions size.
   *
   * @return Total size spanned by indices
   */
  RAJA_INLINE RAJA_HOST_DEVICE static constexpr IdxLin size()
  {
    // Multiply together all of the sizes,
    // replacing 1 for any zero-sized dimensions
    return s_size;
  }

  /*!
   * Given a linear-space index, compute the n-dimensional indices defined
   * by this layout.
   *
   * Note that this operation requires 2n integer divide instructions
   *
   * @param linear_index  Linear space index to be converted to indices.
   * @param indices  Variadic list of indices to be assigned, number must match
   *                 dimensionality of this layout.
   */
  template <typename... Indices>
  RAJA_INLINE RAJA_HOST_DEVICE static void toIndices(IdxLin linear_index,
                                                     Indices &&... indices)
  {
    static_assert(n_dims == sizeof...(Indices),
        "Error: wrong number of indices");
    toIndicesHelper<stride_one_dim, stride_max_dim>(
        linear_index, strides{}, sizes{},
        std::forward<Indices>(indices)...);
  }

  template<camp::idx_t DIM>
  RAJA_INLINE RAJA_HOST_DEVICE static constexpr IndexLinear get_dim_stride()
  {
    return camp::seq_at<DIM, strides>::value;
  }

protected:

  /*!
   * @internal
   *
   * Helper that uses the non-typed toIndices() function, and converts the
   * result to typed indices
   *
   */
  template <camp::idx_t stride_one_dim, camp::idx_t stride_max_dim,
            IdxLin... InvStrides, IdxLin... InvMods, typename... Indices>
  RAJA_INLINE RAJA_HOST_DEVICE static void
  toIndicesHelper(IdxLin linear_index,
                  camp::int_seq<IdxLin, InvStrides...>,
                  camp::int_seq<IdxLin, InvMods...>,
                  Indices&&... indices)
  {
    static_assert(n_dims == sizeof...(Indices),
        "Error: wrong number of indices");
    camp::sink((indices = (camp::decay<Indices>)(
      to_index_calculator<RangeInts, stride_one_dim, stride_max_dim, IdxLin>{}(
          linear_index, InvStrides, InvMods) ))...);
  }
};

template <typename IdxLin, typename Range, typename Sizes, typename Strides>
struct StaticLayoutBase_impl;

template <typename IdxLin,
          IdxLin... RangeInts,
          IdxLin... Sizes,
          IdxLin... Strides>
struct StaticLayoutBase_impl<IdxLin,
                             camp::int_seq<IdxLin, RangeInts...>,
                             camp::int_seq<IdxLin, Sizes...>,
                             camp::int_seq<IdxLin, Strides...>>
    : StaticLayoutNoProjBase_impl<IdxLin,
                                  camp::int_seq<IdxLin, RangeInts...>,
                                  camp::int_seq<IdxLin, Sizes...>,
                                  camp::int_seq<IdxLin, Strides...>>
{
  using Self = StaticLayoutBase_impl<IdxLin,
                             camp::int_seq<IdxLin, RangeInts...>,
                             camp::int_seq<IdxLin, Sizes...>,
                             camp::int_seq<IdxLin, Strides...>>;
  using Base = StaticLayoutNoProjBase_impl<IdxLin,
                                  camp::int_seq<IdxLin, RangeInts...>,
                                  camp::int_seq<IdxLin, Sizes...>,
                                  camp::int_seq<IdxLin, Strides...>>;

  using typename Base::IndexRange;
  using typename Base::StrippedIndexLinear;
  using typename Base::IndexLinear;
  using typename Base::DimTuple;
  using typename Base::DimArr;

  using Base::n_dims;
  using Base::stride_one_dim;
  // always do mod with projection on max size dimension
  static constexpr camp::idx_t stride_max_dim = -1;

  using typename Base::sizes;
  using typename Base::strides;
  using inv_strides = camp::int_seq<IdxLin, (Strides ? Strides : IdxLin(1))...>;
  using inv_mods = camp::int_seq<IdxLin, (Sizes ? Sizes : IdxLin(1))...>;

  /*!
   * Default constructor.
   */
  RAJA_INLINE RAJA_HOST_DEVICE constexpr StaticLayoutBase_impl() : Base() {}

  // Multiply together all of the sizes,
  // replacing 1 for any zero-sized dimensions
  static constexpr IdxLin s_size =
      RAJA::product<IdxLin>((Sizes == IdxLin(0) ? IdxLin(1) : Sizes)...);

  /*!
   * Computes a total size of the layout's space.
   * This is the produce of each dimensions size.
   *
   * @return Total size spanned by indices
   */
  RAJA_INLINE RAJA_HOST_DEVICE static constexpr IdxLin size()
  {
    // Multiply together all of the sizes,
    // replacing 1 for any zero-sized dimensions
    return s_size;
  }

  /*!
   * Given a linear-space index, compute the n-dimensional indices defined
   * by this layout.
   *
   * Note that this operation requires 2n integer divide instructions
   *
   * @param linear_index  Linear space index to be converted to indices.
   * @param indices  Variadic list of indices to be assigned, number must match
   *                 dimensionality of this layout.
   */
  template <typename... Indices>
  RAJA_INLINE RAJA_HOST_DEVICE static void toIndices(IdxLin linear_index,
                                                     Indices&&... indices)
  {
    static_assert(n_dims == sizeof...(Indices),
        "Error: wrong number of indices");
    Base::template toIndicesHelper<stride_one_dim, stride_max_dim>(
        linear_index, inv_strides{}, inv_mods{},
        std::forward<Indices>(indices)...);
  }
};


template <typename LayoutBase, typename IdxLin, typename DimTypeList>
struct TypedStaticLayoutImpl;

template <typename LayoutBase, typename IdxLin, typename... DimTypes>
struct TypedStaticLayoutImpl<LayoutBase, IdxLin, camp::list<DimTypes...>>
    : LayoutBase
{
  using Self = TypedStaticLayoutImpl<LayoutBase, IdxLin, camp::list<DimTypes...>>;
  using Base = LayoutBase;

  using typename Base::IndexRange;
  using StrippedIndexLinear = strip_index_type_t<IdxLin>;
  using IndexLinear = IdxLin;
  using DimTuple = camp::tuple<DimTypes...>;
  using typename Base::DimArr;

  using Base::n_dims;
  using Base::stride_one_dim;
  using Base::stride_max_dim;

  static_assert(n_dims == sizeof...(DimTypes),
      "Error: number of dimension types does not match base layout");
  static_assert(std::is_same<StrippedIndexLinear, typename Base::IndexLinear>::value,
      "Error: linear index types does not match base layout");

  using typename Base::sizes;
  using typename Base::strides;

  using Base::Base;


  /*!
   * Computes a linear space index from specified indices.
   * This is formed by the dot product of the indices and the layout strides.
   *
   * @param indices  Indices in the n-dimensional space of this layout
   * @return Linear space index.
   */
  RAJA_INLINE RAJA_HOST_DEVICE static constexpr IdxLin s_oper(
      DimTypes... indices)
  {
    return IdxLin(Base::s_oper(stripIndexType(indices)...));
  }
  ///
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin operator()(
      DimTypes... indices) const
  {
    return IdxLin(Base::s_oper(stripIndexType(indices)...));
  }

  using Base::s_size;

  /*!
   * Given a linear-space index, compute the n-dimensional indices defined
   * by this layout.
   *
   * Note that this operation requires 2n integer divide instructions
   *
   * @param linear_index  Linear space index to be converted to indices.
   * @param indices  Variadic list of indices to be assigned, number must match
   *                 dimensionality of this layout.
   */
  RAJA_INLINE RAJA_HOST_DEVICE static void toIndices(IdxLin linear_index,
                                                     DimTypes&... indices)
  {
    toTypedIndicesHelper(IndexRange{},
                         std::forward<IdxLin>(linear_index),
                         std::forward<DimTypes &>(indices)...);
  }

private:
  /*!
   * @internal
   *
   * Helper that uses the non-typed toIndices() function, and converts the
   * result to typed indices
   *
   */
  template <camp::idx_t... RangeInts>
  RAJA_INLINE RAJA_HOST_DEVICE static void toTypedIndicesHelper(camp::idx_seq<RangeInts...>,
                                                                IdxLin linear_index,
                                                                DimTypes&... indices)
  {
    StrippedIndexLinear locals[n_dims];
    Base::toIndices(stripIndexType(linear_index), locals[RangeInts]...);
    camp::sink( (indices = static_cast<DimTypes>(locals[RangeInts]))... );
  }
};


template <typename Perm, typename IdxLin, typename Sizes, typename Indexes>
struct StaticLayoutNoProjMaker
{
  using strides = typename detail::StrideCalculatorNoProj<IdxLin, Indexes, Perm, Sizes>::strides;
  using type = StaticLayoutNoProjBase_impl<IdxLin, Indexes, Sizes, strides>;
};

template <typename Perm, typename IdxLin, typename Sizes, typename Indexes>
struct StaticLayoutMaker
{
  using strides = typename detail::StrideCalculator<IdxLin, Indexes, Perm, Sizes>::strides;
  using type = StaticLayoutBase_impl<IdxLin, Indexes, Sizes, strides>;
};

}  // namespace detail


template <typename Perm, typename IdxLin, camp::idx_t... Sizes>
using StaticLayoutNoProjT = typename detail::StaticLayoutNoProjMaker<
    Perm,
    IdxLin,
    camp::int_seq<IdxLin, Sizes...>,
    camp::make_int_seq_t<IdxLin, sizeof...(Sizes)>
    >::type;

template <typename Perm, typename IdxLin, camp::idx_t... Sizes>
using StaticLayoutT = typename detail::StaticLayoutMaker<
    Perm,
    IdxLin,
    camp::int_seq<IdxLin, Sizes...>,
    camp::make_int_seq_t<IdxLin, sizeof...(Sizes)>
    >::type;


template <typename Perm, camp::idx_t... Sizes>
using StaticLayoutNoProj = StaticLayoutNoProjT<Perm, camp::idx_t, Sizes...>;

template <typename Perm, camp::idx_t... Sizes>
using StaticLayout = StaticLayoutT<Perm, camp::idx_t, Sizes...>;


template <typename Perm, typename IdxLin, typename TypeList, camp::idx_t... Sizes>
using TypedStaticLayoutNoProj =
    detail::TypedStaticLayoutImpl<StaticLayoutNoProjT<Perm, strip_index_type_t<IdxLin>, Sizes...>, IdxLin, TypeList>;

template <typename Perm, typename IdxLin, typename TypeList, camp::idx_t... Sizes>
using TypedStaticLayout =
    detail::TypedStaticLayoutImpl<StaticLayoutT<Perm, strip_index_type_t<IdxLin>, Sizes...>, IdxLin, TypeList>;


}  // namespace RAJA

#endif
