/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining offset layout operations for
 *          forallN templates.
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

#ifndef RAJA_OFFSETLAYOUT_HPP
#define RAJA_OFFSETLAYOUT_HPP

#include "RAJA/config.hpp"
#include "RAJA/index/IndexValue.hpp"
#include "RAJA/internal/LegacyCompatibility.hpp"
#include "RAJA/util/Permutations.hpp"
#include "RAJA/util/PermutedLayout.hpp"

#include <array>
#include <limits>
#include "camp/camp.hpp"

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
  using Base = detail::LayoutBase_impl<IndexRange, IdxLin>;
  Base base_;

  IdxLin offsets[sizeof...(RangeInts)];

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

  template <typename... Indices>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin operator()(
      Indices... indices) const
  {
    return base_((indices - offsets[RangeInts])...);
  }

  static RAJA_INLINE OffsetLayout_impl<IndexRange, IdxLin>
  from_layout_and_offsets(
      const std::array<IdxLin, sizeof...(RangeInts)>& offsets_in,
      const Layout<sizeof...(RangeInts), IdxLin>& rhs)
  {
    OffsetLayout_impl ret{rhs};
    VarOps::ignore_args((ret.offsets[RangeInts] = offsets_in[RangeInts])...);
    return ret;
  }

private:
  constexpr RAJA_INLINE RAJA_HOST_DEVICE
  OffsetLayout_impl(const Layout<sizeof...(RangeInts), IdxLin>& rhs)
      : base_{rhs}
  {
  }
};

}  // end internal namespace

template <size_t n_dims = 1, typename IdxLin = Index_type>
struct OffsetLayout
    : public internal::OffsetLayout_impl<camp::make_idx_seq_t<n_dims>, IdxLin> {
  using parent =
      internal::OffsetLayout_impl<camp::make_idx_seq_t<n_dims>, IdxLin>;

  using internal::OffsetLayout_impl<camp::make_idx_seq_t<n_dims>,
                                    IdxLin>::OffsetLayout_impl;

  constexpr RAJA_INLINE RAJA_HOST_DEVICE OffsetLayout(
      const internal::OffsetLayout_impl<camp::make_idx_seq_t<n_dims>, IdxLin>&
          rhs)
      : parent{rhs}
  {
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
