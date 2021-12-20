/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining Layout, a N-dimensional index calculator
 *          with offset indices
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_OFFSETLAYOUT_HPP
#define RAJA_OFFSETLAYOUT_HPP

#include "RAJA/config.hpp"

#include <array>
#include <limits>

#include "camp/camp.hpp"

#include "RAJA/index/IndexValue.hpp"

#include "RAJA/util/Permutations.hpp"
#include "RAJA/util/PermutedLayout.hpp"

namespace RAJA
{

namespace internal
{

template <typename LayoutBase>
struct OffsetLayout_impl
    : LayoutBase
{
  using Base = LayoutBase;
  using Self = OffsetLayout_impl<LayoutBase>;

  using typename Base::IndexRange;
  using typename Base::StrippedIndexLinear;
  using typename Base::IndexLinear;
  using typename Base::DimTuple;
  using typename Base::DimArr;

  using Base::n_dims;
  using Base::limit;
  using Base::stride_one_dim;
  using Base::stride_max_dim;

  using Base::sizes;
  using Base::strides;

  StrippedIndexLinear offsets[n_dims]={0}; //If not specified set to zero

  static RAJA_INLINE Self from_layout_and_offsets(
      const std::array<StrippedIndexLinear, n_dims>& offsets_in,
      const Base& rhs)
  {
    Self ret{rhs};
    ret.shift(offsets_in);
    return ret;
  }

  constexpr RAJA_INLINE OffsetLayout_impl(
      std::array<StrippedIndexLinear, n_dims> lower,
      std::array<StrippedIndexLinear, n_dims> upper)
    : Self(IndexRange{}, lower, upper)
  {
  }

  constexpr RAJA_INLINE RAJA_HOST_DEVICE OffsetLayout_impl(const Base& rhs)
    : Base{rhs}
  {
  }

  constexpr RAJA_INLINE RAJA_HOST_DEVICE OffsetLayout_impl(Self const& c)
    : Self(IndexRange{}, c)
  {
  }

  void shift(const std::array<StrippedIndexLinear, n_dims>& shift)
  {
    for (size_t i=0; i<n_dims; ++i) {
      offsets[i] += shift[i];
    }
  }

  template<camp::idx_t N, typename Idx>
  RAJA_INLINE RAJA_HOST_DEVICE void BoundsCheckError(Idx idx) const
  {
    printf("Error at index %d, value %ld is not within bounds [%ld, %ld] \n",
           static_cast<int>(N), static_cast<long int>(idx),
           static_cast<long int>(offsets[N]), static_cast<long int>(offsets[N] + sizes[N] - 1));
    RAJA_ABORT_OR_THROW("Out of bounds error \n");
  }

  template <camp::idx_t N>
  RAJA_INLINE RAJA_HOST_DEVICE void BoundsCheck() const
  {
  }

  template <camp::idx_t N, typename Idx, typename... Indices>
  RAJA_INLINE RAJA_HOST_DEVICE void BoundsCheck(Idx idx, Indices... indices) const
  {
    if(!(offsets[N] <=idx && idx < offsets[N] + sizes[N]))
    {
      BoundsCheckError<N>(idx);
    }
    RAJA_UNUSED_VAR(idx);
    BoundsCheck<N+1>(indices...);
  }

  template <typename... Indices>
  RAJA_INLINE RAJA_HOST_DEVICE RAJA_BOUNDS_CHECK_constexpr IndexLinear operator()(
      Indices... indices) const
  {
    static_assert(n_dims == sizeof...(Indices),
        "Error: wrong number of indices");
#if defined (RAJA_BOUNDS_CHECK_INTERNAL)
    BoundsCheck<0>(indices...);
#endif
    return callHelper(IndexRange{}, indices...);
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
  RAJA_INLINE RAJA_HOST_DEVICE void toIndices(IndexLinear linear_index,
                                              Indices&&... indices) const
  {
    static_assert(n_dims == sizeof...(Indices),
        "Error: wrong number of indices");
    toIndicesHelper(IndexRange{},
                    std::forward<IndexLinear>(linear_index),
                    std::forward<Indices>(indices)...);
  }

protected:

  template <camp::idx_t... RangeInts>
  constexpr RAJA_INLINE OffsetLayout_impl(
      camp::idx_seq<RangeInts...>,
      std::array<StrippedIndexLinear, n_dims> lower,
      std::array<StrippedIndexLinear, n_dims> upper)
    : Base{(upper[RangeInts] - lower[RangeInts] + 1)...},
      offsets{lower[RangeInts]...}
  {
  }

  template <camp::idx_t... RangeInts>
  constexpr RAJA_INLINE RAJA_HOST_DEVICE OffsetLayout_impl(
      camp::idx_seq<RangeInts...>,
      Self const& c)
    : Base(static_cast<Base const&>(c))
    , offsets{c.offsets[RangeInts]...}
  {
  }

  /*!
   * @internal
   *
   * Helper that calls the base operator() method with offset indices
   *
   */
  template <typename... Indices, camp::idx_t... RangeInts>
  RAJA_INLINE RAJA_HOST_DEVICE IndexLinear callHelper(camp::idx_seq<RangeInts...>,
                                                      Indices... indices) const
  {
    return Base::operator()((indices - offsets[RangeInts])...);
  }

  /*!
   * @internal
   *
   * Helper that uses the base toIndices() method
   *
   */
  template <typename... Indices, camp::idx_t... RangeInts>
  RAJA_INLINE RAJA_HOST_DEVICE void toIndicesHelper(camp::idx_seq<RangeInts...>,
                                                    IndexLinear linear_index,
                                                    Indices&&... indices) const
  {
    static_assert(n_dims == sizeof...(Indices),
        "Error: wrong number of indices");
    StrippedIndexLinear locals[n_dims];
    Base::toIndices(linear_index, locals[RangeInts]...);
    camp::sink( (indices = (camp::decay<Indices>)(
        camp::get<RangeInts>(locals) + offsets[RangeInts]))... );
  }
};

template <typename IdxLin, typename DimTuple, typename LayoutBase>
struct TypedOffsetLayout_impl;

template <typename IdxLin, typename... DimTypes, typename LayoutBase>
struct TypedOffsetLayout_impl<IdxLin, camp::tuple<DimTypes...>, LayoutBase>
    : OffsetLayout_impl<LayoutBase>
{
  using Base = OffsetLayout_impl<LayoutBase>;
  using Self = TypedOffsetLayout_impl<IdxLin, camp::tuple<DimTypes...>, LayoutBase>;

  using typename Base::IndexRange;
  using StrippedIndexLinear = strip_index_type_t<IdxLin>;
  using IndexLinear = IdxLin;
  using DimTuple = camp::tuple<DimTypes...>;
  using typename Base::DimArr;

  using Base::n_dims;
  using Base::limit;
  using Base::stride_one_dim;
  using Base::stride_max_dim;

  static_assert(n_dims == sizeof...(DimTypes),
      "Error: number of dimension types does not match base layout");
  static_assert(std::is_same<StrippedIndexLinear, typename Base::IndexLinear>::value,
      "Error: linear index types does not match base layout");

  // Pull in base constructors
  using Base::Base;

  // consider freeing this function
  // then this and TypedLayoutBase_impl are identical so we could remove this
  static RAJA_INLINE Self from_layout_and_offsets(
      const std::array<StrippedIndexLinear, n_dims>& offsets_in,
      const LayoutBase& rhs)
  {
    Self ret{rhs};
    ret.shift(offsets_in);
    return ret;
  }

  RAJA_INLINE RAJA_HOST_DEVICE RAJA_BOUNDS_CHECK_constexpr IdxLin operator()(
      DimTypes... indices) const
  {
    return IdxLin(Base::operator()(stripIndexType(indices)...));
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
  RAJA_INLINE RAJA_HOST_DEVICE void toIndices(IdxLin linear_index,
                                              DimTypes&... indices) const
  {
    toTypedIndicesHelper(IndexRange{},
                         std::forward<IdxLin>(linear_index),
                         std::forward<DimTypes&>(indices)...);
  }

private:
  /*!
   * @internal
   *
   * Helper that uses the base toIndices() method
   *
   */
  template <camp::idx_t... RangeInts>
  RAJA_INLINE RAJA_HOST_DEVICE void toTypedIndicesHelper(camp::idx_seq<RangeInts...>,
                                                         IdxLin linear_index,
                                                         DimTypes&... indices) const
  {
    StrippedIndexLinear locals[n_dims];
    Base::toIndices(stripIndexType(linear_index), locals[RangeInts]...);
    camp::sink( (indices = static_cast<DimTypes>(locals[RangeInts]))... );
  }
};

}  // namespace internal


template <size_t n_dims = 1, typename IdxLin = Index_type,
          ptrdiff_t StrideOneDim = -1, ptrdiff_t StrideMaxDim = -1>
using OffsetLayoutNoProj = internal::OffsetLayout_impl<
    LayoutNoProj<n_dims, IdxLin, StrideOneDim, StrideMaxDim> >;

template <size_t n_dims = 1, typename IdxLin = Index_type,
          ptrdiff_t StrideOneDim = -1, ptrdiff_t StrideMaxDim = -1>
using OffsetLayout = internal::OffsetLayout_impl<
    Layout<n_dims, IdxLin, StrideOneDim, StrideMaxDim> >;


template <typename IdxLin, typename DimTuple,
          ptrdiff_t StrideOneDim = -1, ptrdiff_t StrideMaxDim = -1>
using TypedOffsetLayoutNoProj = internal::TypedOffsetLayout_impl<
    IdxLin, DimTuple, LayoutNoProj<camp::tuple_size<DimTuple>::value,
                                   strip_index_type_t<IdxLin>,
                                   StrideOneDim, StrideMaxDim> >;

template <typename IdxLin, typename DimTuple,
          ptrdiff_t StrideOneDim = -1, ptrdiff_t StrideMaxDim = -1>
using TypedOffsetLayout = internal::TypedOffsetLayout_impl<
    IdxLin, DimTuple, Layout<camp::tuple_size<DimTuple>::value,
                             strip_index_type_t<IdxLin>,
                             StrideOneDim, StrideMaxDim> >;


template <size_t n_dims, typename IdxLin = Index_type,
          ptrdiff_t StrideOneDim = -1, ptrdiff_t StrideMaxDim = -1>
auto make_offset_layout(const std::array<IdxLin, n_dims>& lower,
                        const std::array<IdxLin, n_dims>& upper)
    -> OffsetLayout<n_dims, IdxLin, StrideOneDim, StrideMaxDim>
{
  return OffsetLayout<n_dims, IdxLin, StrideOneDim, StrideMaxDim>{lower, upper};
}

template <size_t n_dims, typename IdxLin = Index_type,
          ptrdiff_t StrideOneDim = -1, ptrdiff_t StrideMaxDim = -1>
auto make_permuted_offset_layout(const std::array<IdxLin, n_dims>& lower,
                                 const std::array<IdxLin, n_dims>& upper,
                                 const std::array<IdxLin, n_dims>& permutation)
    -> OffsetLayout<n_dims, IdxLin, StrideOneDim, StrideMaxDim>
{
  std::array<IdxLin, n_dims> sizes;
  for (size_t i = 0; i < n_dims; ++i) {
    sizes[i] = upper[i] - lower[i] + 1;
  }
  return OffsetLayout<n_dims, IdxLin, StrideOneDim, StrideMaxDim>::
      from_layout_and_offsets(lower,
          make_permuted_layout<n_dims, IdxLin, StrideOneDim, StrideMaxDim>(
              sizes, permutation));
}

}  // namespace RAJA

#endif
