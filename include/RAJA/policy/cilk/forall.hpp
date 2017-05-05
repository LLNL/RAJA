/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index set and segment iteration
 *          template methods for Intel Cilk Plus execution.
 *
 *          These methods work only on platforms that support Cilk Plus.
 *
 ******************************************************************************
 */

#ifndef RAJA_forall_cilk_HXX
#define RAJA_forall_cilk_HXX

#include "RAJA/config.hpp"
#include "RAJA/policy/cilk/policy.hpp"

#if defined(RAJA_ENABLE_CILK)

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

#include "RAJA/util/types.hpp"

#include "RAJA/internal/fault_tolerance.hpp"

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

namespace RAJA
{

namespace impl
{

//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over iterables.
//
//////////////////////////////////////////////////////////////////////
//

template <typename Iterable, typename Func>
RAJA_INLINE void forall(const cilk_for_exec &,
                        Iterable &&iter,
                        Func &&loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
  cilk_for(Index_type i = 0; i < distance; ++i) { loop_body(begin[i]); }
}

template <typename Iterable, typename Func>
RAJA_INLINE void forall_Icount(const cilk_for_exec &,
                               Iterable &&iter,
                               Index_type icount,
                               Func &&loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
  cilk_for(Index_type i = 0; i < distance; ++i)
  {
    loop_body(i + icount, begin[i]);
  }
}

}  // closing brace for impl namespace

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_CILK guard

#endif  // closing endif for header file include guard
