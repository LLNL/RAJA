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
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
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

template <typename Range, typename IdxLin>
struct OffsetLayout_impl;

template <camp::idx_t... RangeInts, typename IdxLin>
struct OffsetLayout_impl<camp::idx_seq<RangeInts...>, IdxLin> {
  using Self = OffsetLayout_impl<camp::idx_seq<RangeInts...>, IdxLin>;
  using IndexRange = camp::idx_seq<RangeInts...>;
  using IndexLinear = IdxLin;
  using Base = RAJA::detail::LayoutBase_impl<IndexRange, IdxLin>;
  Base base_;

  static constexpr camp::idx_t stride_one_dim = Base::stride_one_dim;

  static constexpr size_t n_dims = sizeof...(RangeInts);
  IdxLin offsets[n_dims]={0}; //If not specified set to zero

  constexpr RAJA_INLINE OffsetLayout_impl(
      std::array<IdxLin, sizeof...(RangeInts)> lower,
      std::array<IdxLin, sizeof...(RangeInts)> upper)
      : base_{(upper[RangeInts] - lower[RangeInts] + 1)...},
        offsets{lower[RangeInts]...}
  {
  }

  constexpr RAJA_INLINE RAJA_HOST_DEVICE OffsetLayout_impl(Self const& c)
      : base_(c.base_), offsets{c.offsets[RangeInts]...}
  {
  }

  void shift(std::array<IdxLin, sizeof...(RangeInts)> shift)
  {
    for(size_t i=0; i<n_dims; ++i) offsets[i] += shift[i];
  }

  template<camp::idx_t N, typename Idx>
  RAJA_INLINE RAJA_HOST_DEVICE void BoundsCheckError(Idx idx) const
  {
    printf("Error at index %d, value %ld is not within bounds [%ld, %ld] \n",
           static_cast<int>(N), static_cast<long int>(idx),
           static_cast<long int>(offsets[N]), static_cast<long int>(offsets[N] + base_.sizes[N] - 1));
    RAJA_ABORT_OR_THROW("Out of bounds error \n");
  }

  template <camp::idx_t N>
  RAJA_INLINE RAJA_HOST_DEVICE void BoundsCheck() const
  {
  }

  template <camp::idx_t N, typename Idx, typename... Indices>
  RAJA_INLINE RAJA_HOST_DEVICE void BoundsCheck(Idx idx, Indices... indices) const
  {
    if(!(offsets[N] <=idx && idx < offsets[N] + base_.sizes[N]))
    {
      BoundsCheckError<N>(idx);
    }
    RAJA_UNUSED_VAR(idx);
    BoundsCheck<N+1>(indices...);
  }

  template <typename... Indices>
  RAJA_INLINE RAJA_HOST_DEVICE RAJA_BOUNDS_CHECK_constexpr IdxLin operator()(
      Indices... indices) const
  {
#if defined (RAJA_BOUNDS_CHECK_INTERNAL)
    BoundsCheck<0>(indices...);
#endif
    return base_((indices - offsets[RangeInts])...);
  }

  static RAJA_INLINE OffsetLayout_impl<IndexRange, IdxLin>
  from_layout_and_offsets(
      const std::array<IdxLin, sizeof...(RangeInts)>& offsets_in,
      const Layout<sizeof...(RangeInts), IdxLin>& rhs)
  {
    OffsetLayout_impl ret{rhs};
    camp::sink((ret.offsets[RangeInts] = offsets_in[RangeInts])...);
    return ret;
  }

  constexpr RAJA_INLINE RAJA_HOST_DEVICE
  OffsetLayout_impl(const Layout<sizeof...(RangeInts), IdxLin>& rhs)
      : base_{rhs}
  {
  }

  template<camp::idx_t DIM>
  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr
  IndexLinear get_dim_stride() const {
    return base_.get_dim_stride();
  }
};

}  // namespace internal

template <size_t n_dims = 1, typename IdxLin = Index_type>
struct OffsetLayout
    : public internal::OffsetLayout_impl<camp::make_idx_seq_t<n_dims>, IdxLin> {
  using Base =
      internal::OffsetLayout_impl<camp::make_idx_seq_t<n_dims>, IdxLin>;

  using internal::OffsetLayout_impl<camp::make_idx_seq_t<n_dims>,
                                    IdxLin>::OffsetLayout_impl;

  constexpr RAJA_INLINE RAJA_HOST_DEVICE OffsetLayout(
      const internal::OffsetLayout_impl<camp::make_idx_seq_t<n_dims>, IdxLin>&
          rhs)
      : Base{rhs}
  {
  }
};

//TypedOffsetLayout
template <typename IdxLin, typename DimTuple>
struct TypedOffsetLayout;

template <typename IdxLin, typename... DimTypes>
struct TypedOffsetLayout<IdxLin, camp::tuple<DimTypes...>>
: public OffsetLayout<sizeof...(DimTypes), Index_type>
{
   using Self = TypedOffsetLayout<IdxLin, camp::tuple<DimTypes...>>;
   using Base = OffsetLayout<sizeof...(DimTypes), Index_type>;
   using DimArr = std::array<Index_type, sizeof...(DimTypes)>;
   using IndexLinear = IdxLin;

   // Pull in base coonstructors
 #if 0
   // This breaks with nvcc11
 using Base::Base;
 #else
   using OffsetLayout<sizeof...(DimTypes), Index_type>::OffsetLayout;
 #endif

  RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin operator()(DimTypes... indices) const
  {
    return IdxLin(Base::operator()(stripIndexType(indices)...));
  }

};


template <size_t n_dims, typename IdxLin = Index_type>
auto make_offset_layout(const std::array<IdxLin, n_dims>& lower,
                        const std::array<IdxLin, n_dims>& upper)
    -> OffsetLayout<n_dims, IdxLin>
{
  return OffsetLayout<n_dims, IdxLin>{lower, upper};
}

template <size_t Rank, typename IdxLin = Index_type>
auto make_permuted_offset_layout(const std::array<IdxLin, Rank>& lower,
                                 const std::array<IdxLin, Rank>& upper,
                                 const std::array<IdxLin, Rank>& permutation)
    -> decltype(make_offset_layout<Rank, IdxLin>(lower, upper))
{
  std::array<IdxLin, Rank> sizes;
  for (size_t i = 0; i < Rank; ++i) {
    sizes[i] = upper[i] - lower[i] + 1;
  }
  return internal::OffsetLayout_impl<camp::make_idx_seq_t<Rank>, IdxLin>::
      from_layout_and_offsets(lower, make_permuted_layout(sizes, permutation));
}

}  // namespace RAJA

#endif
