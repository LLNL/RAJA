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

#include "RAJA/index/IndexValue.hpp"
#include "RAJA/internal/LegacyCompatibility.hpp"
#include "RAJA/util/Layout.hpp"
#include "RAJA/util/Permutations.hpp"
#include "RAJA/util/Operators.hpp"

namespace RAJA
{

template <size_t Rank, typename IdxLin = Index_type>
auto make_permuted_layout(std::array<IdxLin, Rank> sizes,
                          std::array<size_t, Rank> permutation) ->
Layout<Rank, IdxLin>
{
  std::array<IdxLin, Rank> strides, mods;
  std::array<IdxLin, Rank> folded_strides, lmods;
  for (size_t i = 0; i < Rank; ++i) {
    folded_strides[i] = 1;
    for (size_t j = 0; j < i; ++j) {
      folded_strides[j] *= sizes[permutation[i]];
    }
  }

  for (size_t i = 0; i < Rank; ++i) {
    strides[permutation[i]] = folded_strides[i];
  }

  for (size_t i = 1; i < Rank; i++) {
    lmods[i] = folded_strides[i - 1];
  }
  lmods[0] = RAJA::operators::limits<IdxLin>::max();

  for (size_t i = 0; i < Rank; ++i) {
    mods[permutation[i]] = lmods[i];
  }

  return Layout<Rank, IdxLin>(sizes, strides, mods);
}


template<size_t ... Ints>
using Perm = VarOps::index_sequence<Ints...>;
template<size_t N>
using MakePerm = VarOps::make_index_sequence<N>;

}  // namespace RAJA

#endif
