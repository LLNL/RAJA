/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index set and segment iteration
 *          template methods for sequential execution.
 *
 *          These methods should work on any platform.
 *          
 *          Note: GNU compiler does not enforce sequential iterations.
 *
 ******************************************************************************
 */

#ifndef RAJA_forall_sequential_HPP
#define RAJA_forall_sequential_HPP

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

#include "RAJA/config.hpp"

#include "RAJA/util/types.hpp"

#include "RAJA/policy/sequential/policy.hpp"

#include "RAJA/index/ListSegment.hpp"
#include "RAJA/index/RangeSegment.hpp"

#include "RAJA/internal/fault_tolerance.hpp"

using RAJA::concepts::enable_if;

namespace RAJA
{

namespace impl
{


//
//////////////////////////////////////////////////////////////////////
//
// The following function templates iterate over index set segments
// sequentially.  Segment execution is defined by segment
// execution policy template parameter.
//
//////////////////////////////////////////////////////////////////////
//

template <typename Iterable, typename Func>
RAJA_INLINE void forall(const seq_exec &, Iterable &&iter, Func &&body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);

  RAJA_NO_SIMD
  for (decltype(distance) i = 0; i < distance; ++i) {
    body(*(begin + i));
  }
}

template <typename Iterable, typename Func, typename IndexType>
RAJA_INLINE concepts::enable_if<type_traits::is_integral<IndexType>>
forall_Icount(const seq_exec &, Iterable &&iter, IndexType icount, Func &&body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);

  RAJA_NO_SIMD
  for (decltype(distance) i = 0; i < distance; ++i) {
    body(static_cast<IndexType>(i + icount), *(begin + i));
  }
}

}  // closing brace for impl namespace

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
