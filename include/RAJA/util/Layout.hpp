/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining layout operations for forallN templates.
 *
 ******************************************************************************
 */

#ifndef RAJA_LAYOUT_HXX__
#define RAJA_LAYOUT_HXX__

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

namespace RAJA
{

template <typename Range, typename IdxLin = Index_type>
struct LayoutBase_impl { };

template <size_t... RangeInts,
        typename IdxLin>
struct LayoutBase_impl<VarOps::index_sequence<RangeInts...>,
        IdxLin> {
public:
  typedef IdxLin IndexLinear;
  typedef VarOps::make_index_sequence<sizeof...(RangeInts)> IndexRange;

  static constexpr size_t n_dims = sizeof...(RangeInts);
  static constexpr size_t limit = RAJA::operators::limits<IdxLin>::max();

  // const char *index_types[sizeof...(RangeInts)];

  const IdxLin sizes[n_dims];
  IdxLin strides[n_dims];
  IdxLin mods[n_dims];


  // TODO: this should be constexpr in c++14 mode
  template <typename... Types>
  RAJA_INLINE RAJA_HOST_DEVICE LayoutBase_impl(Types... ns)
  : sizes{convertIndex<IdxLin>(ns)...}
  {
    static_assert(n_dims == sizeof ... (Types), "number of dimensions must match");
    for (size_t i = 0; i < n_dims; i++) {
      strides[i] = 1;
      for (size_t j = 0; j < i; j++) {
        strides[j] *= sizes[i];
      }
    }

    for (size_t i = 1; i < n_dims; i++) {
      mods[i] = strides[i - 1];
    }
    mods[0] = limit;
  }

  constexpr RAJA_INLINE RAJA_HOST_DEVICE LayoutBase_impl(const LayoutBase_impl<IndexRange, IdxLin>& rhs)
  : sizes{rhs.sizes[RangeInts]...},
    strides{rhs.strides[RangeInts]...},
    mods{rhs.mods[RangeInts]...}
  { }

  template <typename... Types>
  constexpr RAJA_INLINE LayoutBase_impl(const std::array<IdxLin, n_dims> &sizes_in,
                                        const std::array<IdxLin, n_dims> &strides_in,
                                        const std::array<IdxLin, n_dims> &mods_in)
  : sizes{sizes_in[RangeInts]...},
    strides{strides_in[RangeInts]...},
    mods{mods_in[RangeInts]...}
  { }

  template<typename... Indices>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin operator()(Indices... indices) const
  {
    return VarOps::sum<IdxLin>((indices * strides[RangeInts])...);
  }

  template<typename... Indices>
  RAJA_INLINE RAJA_HOST_DEVICE void toIndices(IdxLin linear_index,
                                              Indices &... indices) const
  {
    VarOps::ignore_args( (indices = (linear_index / strides[RangeInts]) % sizes[RangeInts])... );
  }
};

template <size_t... RangeInts, typename IdxLin>
constexpr size_t LayoutBase_impl<VarOps::index_sequence<RangeInts...>, IdxLin>::n_dims;
template <size_t... RangeInts, typename IdxLin>
constexpr size_t LayoutBase_impl<VarOps::index_sequence<RangeInts...>, IdxLin>::limit;

template <size_t n_dims, typename IdxLin = Index_type>
using Layout = LayoutBase_impl<VarOps::make_index_sequence<n_dims>, IdxLin>;


}  // namespace RAJA

#endif
