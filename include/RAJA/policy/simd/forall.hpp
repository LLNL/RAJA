/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA segment template methods for
 *          SIMD execution.
 *
 *          These methods should work on any platform. They make no
 *          asumptions about data alignment.
 *
 ******************************************************************************
 */

#ifndef RAJA_forall_simd_HPP
#define RAJA_forall_simd_HPP

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

#include <iterator>
#include <type_traits>

#include "RAJA/config.hpp"

#include "RAJA/util/types.hpp"

#include "RAJA/internal/fault_tolerance.hpp"

#include "RAJA/policy/simd/policy.hpp"

namespace RAJA
{

namespace impl
{

// SIMD forall(ListSegment)
template <typename LSegment, typename Func>
RAJA_INLINE
    typename std::enable_if<std::is_base_of<ListSegment, LSegment>::value>::type
    forall(const simd_exec &, LSegment &&iseg, Func &&loop_body)
{
  const auto *RAJA_RESTRICT idx = iseg.getIndex();
  auto len = iseg.getLength();
  for (decltype(len) k = 0; k < len; ++k) {
    loop_body(idx[k]);
  }
}

// SIMD forall(Iterable)
template <typename Iterable, typename Func>
RAJA_INLINE typename std::enable_if<!std::is_base_of<ListSegment,
                                                     Iterable>::value>::type
forall(const simd_exec &, Iterable &&iter, Func &&loop_body)
{
  // TODO: if KNL, make sure long is used
  auto len = iter.size();
  auto ii = std::begin(iter);
  RAJA_SIMD
  for (decltype(len) i = 0; i < len; ++i) {
    loop_body(*(ii + i));
  }
}

// SIMD forall(ListSegment)
template <typename LSegment, typename IndexType, typename Func>
RAJA_INLINE typename std::enable_if<std::is_integral<IndexType>::value
                                    && std::is_base_of<ListSegment,
                                                       LSegment>::value>::type
forall_Icount(const simd_exec &,
              LSegment &&iseg,
              IndexType icount,
              Func &&loop_body)
{
  const auto *RAJA_RESTRICT idx = iseg.getIndex();
  auto len = iseg.getLength();
  for (decltype(len) k = 0; k < len; ++k) {
    loop_body(static_cast<IndexType>(k + icount), idx[k]);
  }
}

// SIMD forall(Iterable)
template <typename Iterable, typename IndexType, typename Func>
RAJA_INLINE typename std::enable_if<std::is_integral<IndexType>::value
                                    && !std::is_base_of<ListSegment,
                                                        Iterable>::value>::type
forall_Icount(const simd_exec &,
              Iterable &&iter,
              IndexType icount,
              Func &&loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
  RAJA_SIMD
  for (decltype(distance) i = 0; i < distance; ++i) {
    loop_body(static_cast<IndexType>(i + icount), begin[i]);
  }
}

}  // closing brace for impl namespace

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
