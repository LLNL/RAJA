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

#ifndef RAJA_forall_simd_HXX
#define RAJA_forall_simd_HXX

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

#include "RAJA/internal/fault_tolerance.hpp"

#include "RAJA/policy/simd/policy.hpp"

namespace RAJA
{

namespace impl
{

template <typename Iterable, typename Func>
RAJA_INLINE void forall(const simd_exec &, Iterable &&iter, Func &&loop_body)
{
  auto end = std::end(iter);
  RAJA_SIMD
  for (auto ii = std::begin(iter); ii < end; ++ii) {
    loop_body(*ii);
  }
}

template <typename Iterable, typename Func>
RAJA_INLINE void forall_Icount(const simd_exec &,
                               Iterable &&iter,
                               Index_type icount,
                               Func &&loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
  RAJA_SIMD
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(i + icount, begin[i]);
  }
}

//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over list segment objects.
//
// NOTE: These operations will not vectorize. We include them here and
//       force sequential execution for convenience.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  "Fake" SIMD iteration over list segment object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall(simd_exec, const ListSegment &iseg, LOOP_BODY loop_body)
{
  const Index_type *RAJA_RESTRICT idx = iseg.getIndex();
  Index_type len = iseg.getLength();

  RAJA_FT_BEGIN;

  for (Index_type k = 0; k < len; ++k) {
    loop_body(idx[k]);
  }

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  "Fake" SIMD iteration over list segment object with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall_Icount(simd_exec,
                               const ListSegment &iseg,
                               Index_type icount,
                               LOOP_BODY loop_body)
{
  const Index_type *RAJA_RESTRICT idx = iseg.getIndex();
  Index_type len = iseg.getLength();

  RAJA_FT_BEGIN;

  for (Index_type k = 0; k < len; ++k) {
    loop_body(k + icount, idx[k]);
  }

  RAJA_FT_END;
}

//
//////////////////////////////////////////////////////////////////////
//
// SIMD execution policy does not apply to iteration over index
// set segments, only to execution of individual segments. So there
// are no index set traversal methods in this file.
//
//////////////////////////////////////////////////////////////////////
//

}  // closing brace for impl namespace

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
