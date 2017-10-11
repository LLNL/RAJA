/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining layout permutation operations for
 *          forallN templates.
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

#ifndef RAJA_PERMUTEDLAYOUT_HPP
#define RAJA_PERMUTEDLAYOUT_HPP

#include <iostream>

#include "RAJA/config.hpp"
#include "RAJA/index/IndexValue.hpp"
#include "RAJA/internal/LegacyCompatibility.hpp"
#include "RAJA/util/Layout.hpp"
#include "RAJA/util/Operators.hpp"
#include "RAJA/util/Permutations.hpp"

namespace RAJA
{

/*!
 * @brief Creates a permuted Layout object.
 *
 * Allows control over the striding order when creating a Layout.
 *
 * For example:
 *
 *     // Create a layout object with the default striding order
 *     // The indices, left to right, have longest stride to stride-1
 *     Layout<3> layout(5,7,11);
 *
 *     // The above is equivalent to:
 *     Layout<3> default_layout = make_permuted_layout({5,7,11},
 * PERM_IJK::value);
 *
 *     // Create a layout object with permuted order
 *     // In this case, J is stride-1, and K has the longest stride
 *     Layout<3> perm_layout = make_permuted_layout({5,7,11}, PERM_KIJ::value);
 *
 *
 * Permutation of up to rank 5 are provided with PERM_I ... PERM_IJKLM
 * aliases in RAJA/util/Permutations.hpp as a convenience.
 *
 * Since permutations are represented using std::array<size_t, Rank> object,
 * arbitrary rank permutations can be used.
 *
 *
 */
template <size_t Rank, typename IdxLin = Index_type>
auto make_permuted_layout(std::array<IdxLin, Rank> sizes,
                          std::array<camp::idx_t, Rank> permutation)
    -> Layout<Rank, IdxLin>
{
  std::array<IdxLin, Rank> strides;
  std::array<IdxLin, Rank> folded_strides;
  for (size_t i = 0; i < Rank; ++i) {
    // If the size of dimension i is zero, then the stride is zero
    folded_strides[i] = sizes[permutation[i]] ? 1 : 0;
    for (size_t j = i + 1; j < Rank; ++j) {
      folded_strides[i] *= sizes[permutation[j]] ? sizes[permutation[j]] : 1;
    }
  }

  for (size_t i = 0; i < Rank; ++i) {
    strides[permutation[i]] = folded_strides[i];
  }


  return Layout<Rank, IdxLin>(sizes, strides);
}


template <camp::idx_t... Ints>
using Perm = camp::idx_seq<Ints...>;
template <camp::idx_t N>
using MakePerm = typename camp::make_idx_seq<N>::type;

}  // namespace RAJA

#endif
