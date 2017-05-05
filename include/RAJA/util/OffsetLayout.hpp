/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining layout operations for forallN templates.
 *
 ******************************************************************************
 */

#ifndef RAJA_OFFSETLAYOUT_HXX__
#define RAJA_OFFSETLAYOUT_HXX__

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

#include <iostream>
#include <limits>

#include "RAJA/index/IndexValue.hpp"
#include "RAJA/internal/LegacyCompatibility.hpp"
#include "RAJA/util/Permutations.hpp"
#include "RAJA/util/PermutedLayout.hpp"

namespace RAJA
{

namespace internal
{

template <typename Range, typename IdxLin>
struct OffsetLayout_impl;

template <size_t... RangeInts, typename IdxLin>
struct OffsetLayout_impl<VarOps::index_sequence<RangeInts...>, IdxLin> {
  using IndexRange = VarOps::index_sequence<RangeInts...>;
  using Base = LayoutBase_impl<IndexRange, IdxLin>;
  Base base_;

  IdxLin offsets[sizeof...(RangeInts)];

  constexpr RAJA_INLINE
  OffsetLayout_impl(std::array<IdxLin, sizeof...(RangeInts)> lower,
                    std::array<IdxLin, sizeof...(RangeInts)> upper)
      : base_{(upper[RangeInts] - lower[RangeInts] + 1)...},
        offsets{lower[RangeInts]...}
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
    return internal::OffsetLayout_impl<IndexRange, IdxLin>(offsets_in, rhs);
  }

private:
  constexpr RAJA_INLINE
  OffsetLayout_impl(const std::array<IdxLin, sizeof...(RangeInts)>& offsets_in,
                    const Layout<sizeof...(RangeInts), IdxLin>& rhs)
      : base_{rhs}, offsets{offsets_in[RangeInts]...}
  {
  }
};

}  // end internal namespace

template <size_t n_dims = 1, typename IdxLin = Index_type>
struct OffsetLayout
    : public internal::OffsetLayout_impl<VarOps::make_index_sequence<n_dims>,
                                         IdxLin> {
  using parent = internal::OffsetLayout_impl<
      VarOps::make_index_sequence<n_dims>, IdxLin>;

  using internal::OffsetLayout_impl<VarOps::make_index_sequence<n_dims>,
                                    IdxLin>::OffsetLayout_impl;

  constexpr RAJA_INLINE RAJA_HOST_DEVICE OffsetLayout(
      const internal::OffsetLayout_impl<VarOps::make_index_sequence<n_dims>,
                                        IdxLin> &rhs)
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
auto make_permuted_offset_layout(const std::array<IdxLin,
                                                            Rank>&
                                                            lower,
                                           const std::array<IdxLin,
                                                            Rank>&
                                                            upper,
                                           const std::array<size_t,
                                                            Rank>&
                                                            permutation)
    -> decltype(make_offset_layout<Rank, IdxLin>(lower, upper))
{
  std::array<IdxLin, Rank> sizes;
  for (size_t i=0; i < Rank; ++i) {
    sizes[i] = upper[i] - lower[i] + 1;
  }
  return internal::OffsetLayout_impl<VarOps::make_index_sequence<Rank>, IdxLin>::from_layout_and_offsets
      (lower, make_permuted_layout(sizes, permutation));
}

}  // namespace RAJA

#endif
