/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining layout operations for forallN templates.
 *
 ******************************************************************************
 */

#ifndef RAJA_PERMUTEDLAYOUT_HXX__
#define RAJA_PERMUTEDLAYOUT_HXX__

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

#include "RAJA/IndexValue.hxx"
#include "RAJA/LegacyCompatibility.hxx"
#include "RAJA/foralln/Layout.hxx"
#include "RAJA/foralln/Permutations.hxx"

namespace RAJA
{

namespace internal
{
template <typename IdxLin,
          size_t... RangeInts,
          size_t... PermInts,
          typename... Sizes>
auto make_permuted_layout_impl(VarOps::index_sequence<RangeInts...> IndexRange,
                               VarOps::index_sequence<PermInts...> Permutation,
                               Sizes... sizes)
    -> Layout<sizeof...(Sizes), IdxLin>
{
  constexpr int n_dims = sizeof...(sizes);
  constexpr IdxLin limit = std::numeric_limits<IdxLin>::max();

  std::array<IdxLin, n_dims> sizes_out{sizes...};
  std::array<IdxLin, n_dims> strides;
  std::array<IdxLin, n_dims> mods;

  Index_type perms[n_dims];
  VarOps::assign_args(perms, IndexRange, PermInts...);
  Index_type swizzled_sizes[] = {sizes_out[PermInts]...};
  Index_type folded_strides[n_dims];
  for (size_t i = 0; i < n_dims; i++) {
    folded_strides[i] = 1;
    for (size_t j = 0; j < i; j++) {
      folded_strides[j] *= swizzled_sizes[i];
    }
  }
  assign(strides, folded_strides, Permutation, IndexRange);

  Index_type lmods[n_dims];
  for (size_t i = 1; i < n_dims; i++) {
    lmods[i] = folded_strides[i - 1];
  }
  lmods[0] = limit;

  assign(mods, lmods, Permutation, IndexRange);
  return Layout<n_dims, IdxLin>(sizes_out, strides, mods);
}
}

template <typename Permutation, typename IdxLin = Index_type, typename... Sizes>
auto make_permuted_layout(Sizes... sizes) -> Layout<sizeof...(Sizes)>
{
  return internal::make_permuted_layout_impl<IdxLin>(
      VarOps::make_index_sequence<sizeof...(Sizes)>(), Permutation(), sizes...);
}

template <typename Range, typename Perm, typename IdxLin>
struct Layout_impl;
template <size_t... RangeInts, size_t... PermInts, typename IdxLin>
struct Layout_impl<VarOps::index_sequence<RangeInts...>,
                   VarOps::index_sequence<PermInts...>,
                   IdxLin> {
  using Base = LayoutBase_impl<VarOps::index_sequence<RangeInts...>, IdxLin>;
  Base base_;

  typedef VarOps::index_sequence<PermInts...> Permutation;
  typedef VarOps::make_index_sequence<sizeof...(RangeInts)> IndexRange;

  Index_type perms[Base::n_dims];

  // TODO: this should be constexpr in c++14 mode
  template <typename... Types>
  RAJA_INLINE RAJA_HOST_DEVICE Layout_impl(Types... ns) : base_{ns...}
  {
    VarOps::assign_args(perms, IndexRange{}, PermInts...);
    Index_type swizzled_sizes[] = {base_.sizes[PermInts]...};
    Index_type folded_strides[Base::n_dims];
    for (size_t i = 0; i < Base::n_dims; i++) {
      folded_strides[i] = 1;
      for (size_t j = 0; j < i; j++) {
        folded_strides[j] *= swizzled_sizes[i];
      }
    }
    assign(base_.strides, folded_strides, Permutation{}, IndexRange{});

    Index_type lmods[Base::n_dims];
    for (size_t i = 1; i < Base::n_dims; i++) {
      lmods[i] = folded_strides[i - 1];
    }
    lmods[0] = Base::limit;

    assign(base_.mods, lmods, Permutation{}, IndexRange{});
  }

  template <typename... Indices>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin operator()(
      Indices... indices) const
  {
    return base_((indices)...);
  }
};

template <typename Permutation, typename IdxLin = Index_type>
struct PermutedLayout {
};
template <size_t... PermInts, typename IdxLin>
struct PermutedLayout<VarOps::index_sequence<PermInts...>, IdxLin>
    : public Layout_impl<VarOps::make_index_sequence<sizeof...(PermInts)>,
                         VarOps::index_sequence<PermInts...>,
                         IdxLin> {
  using Layout_impl<VarOps::make_index_sequence<sizeof...(PermInts)>,
                    VarOps::index_sequence<PermInts...>,
                    IdxLin>::Layout_impl;
};

}  // namespace RAJA

template <typename... Args>
std::ostream &operator<<(std::ostream &os, RAJA::Layout_impl<Args...> const &m)
{
  os << "permutation:" << m.base_.perms << std::endl;
  os << "sizes:" << m.base_.sizes << std::endl;
  os << "mods:" << m.base_.mods << std::endl;
  os << "strides:" << m.base_.strides << std::endl;
  return os;
}


#endif
