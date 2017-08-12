/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index set and segment iteration
 *          template methods for TBB.
 *
 *          These methods should work on any platform that supports TBB.
 *
 ******************************************************************************
 */

#ifndef RAJA_forall_tbb_HPP
#define RAJA_forall_tbb_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_TBB)

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

#include "RAJA/policy/tbb/policy.hpp"

#include "RAJA/index/IndexSet.hpp"
#include "RAJA/index/ListSegment.hpp"
#include "RAJA/index/RangeSegment.hpp"

#include "RAJA/internal/fault_tolerance.hpp"

#include <tbb/tbb.h>


namespace RAJA
{
#if TBB_VERSION_MAJOR >= 2017
using tbb_static_partitioner = tbb::static_partitioner;
#else
// @trws: This is not static, but it seems to be the least damaging option
// available pre-2017
using tbb_static_partitioner = tbb::auto_partitioner;
#endif

namespace impl
{


///
/// TBB parallel for policy implementation
///

template <typename Iterable, typename Func>
RAJA_INLINE void forall(const tbb_for_dynamic& p,
                        Iterable&& iter,
                        Func&& loop_body)
{
  using brange = tbb::blocked_range<decltype(iter.begin())>;
  tbb::parallel_for(brange(std::begin(iter), std::end(iter), p.grain_size),
                    [=](const brange& r) {
                      for (const auto& i : r)
                        loop_body(i);
                    });
}

template <typename Iterable, typename IndexType, typename Func>
RAJA_INLINE typename std::enable_if<std::is_integral<IndexType>::value>::type
forall_Icount(const tbb_for_dynamic& p,
              Iterable&& iter,
              IndexType icount,
              Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
  using brange = tbb::blocked_range<decltype(distance)>;
  tbb::parallel_for(brange(0, distance, p.grain_size), [=](const brange& r) {
    for (decltype(distance) i = r.begin(); i != r.end(); ++i)
      loop_body(static_cast<IndexType>(i + icount), begin[i]);
  });
}

///
/// TBB parallel for static policy implementation
///

template <typename Iterable, typename Func, size_t ChunkSize>
RAJA_INLINE void forall(const tbb_for_static<ChunkSize>&,
                        Iterable&& iter,
                        Func&& loop_body)
{
  using brange = tbb::blocked_range<decltype(iter.begin())>;
  tbb::parallel_for(brange(std::begin(iter), std::end(iter), ChunkSize),
                    [=](const brange& r) {
                      for (const auto& i : r)
                        loop_body(i);
                    },
                    tbb_static_partitioner{});
}

template <typename Iterable,
          typename IndexType,
          typename Func,
          size_t ChunkSize>
RAJA_INLINE typename std::enable_if<std::is_integral<IndexType>::value>::type
forall_Icount(const tbb_for_static<ChunkSize>&,
              Iterable&& iter,
              IndexType icount,
              Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
  using brange = tbb::blocked_range<decltype(distance)>;
  tbb::parallel_for(brange(0, distance, ChunkSize),
                    [=](const brange& r) {
                      for (decltype(distance) i = r.begin(); i != r.end(); ++i)
                        loop_body(static_cast<IndexType>(i + icount), begin[i]);
                    },
                    tbb_static_partitioner{});
}


}  // closing brace for impl namespace

}  // closing brace for RAJA namespace

#endif  // closing endif for if defined(RAJA_ENABLE_TBB)

#endif  // closing endif for header file include guard
