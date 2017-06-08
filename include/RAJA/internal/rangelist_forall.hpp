/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file with centralized Range and List Segment execution
 *          for IndexSets.
 *
 ******************************************************************************
 */

#ifndef RAJA_rangelist_forall_HPP
#define RAJA_rangelist_forall_HPP

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
#include "RAJA/pattern/forall.hpp"
#include "RAJA/policy/sequential.hpp"

namespace RAJA
{
namespace impl
{

/*!
 ******************************************************************************
 *
 * \brief Execute Range or List segment from forall traversal method.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T, typename LOOP_BODY>
RAJA_INLINE void executeRangeList_forall(const IndexSetSegInfo* seg_info,
                                         LOOP_BODY&& loop_body)
{
  const BaseSegment* iseg = seg_info->getSegment();
  SegmentType segtype = iseg->getType();

  switch (segtype) {
    case _RangeSeg_: {
      const RangeSegment* tseg = static_cast<const RangeSegment*>(iseg);
      impl::forall(SEG_EXEC_POLICY_T(), *tseg, loop_body);
      break;
    }

#if 0  // RDH RETHINK
    case _RangeStrideSeg_ : {
         const RangeStrideSegment* tseg =
            static_cast<const RangeStrideSegment*>(iseg);
         impl::forall(
            SEG_EXEC_POLICY_T(),
            tseg->getBegin(), tseg->getEnd(), tseg->getStride(),
            loop_body
         );
         break;
      }
#endif

    case _ListSeg_: {
      const ListSegment* tseg = static_cast<const ListSegment*>(iseg);
      impl::forall(SEG_EXEC_POLICY_T(), *tseg, loop_body);
      break;
    }

    default: {
    }

  }  // switch on segment type
}

/*!
 ******************************************************************************
 *
 * \brief Execute Range or List segment from forall_Icount traversal method.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T, typename LOOP_BODY>
RAJA_INLINE void executeRangeList_forall_Icount(const IndexSetSegInfo* seg_info,
                                                LOOP_BODY&& loop_body)
{
  const BaseSegment* iseg = seg_info->getSegment();
  SegmentType segtype = iseg->getType();

  Index_type icount = seg_info->getIcount();

  switch (segtype) {
    case _RangeSeg_: {
      const RangeSegment* tseg = static_cast<const RangeSegment*>(iseg);
      impl::forall_Icount(SEG_EXEC_POLICY_T(), *tseg, icount, loop_body);
      break;
    }

#if 0  // RDH RETHINK
    case _RangeStrideSeg_ : {
         const RangeStrideSegment* tseg =
            static_cast<const RangeStrideSegment*>(iseg);
         forall_Icount(
            SEG_EXEC_POLICY_T(),
            tseg->getBegin(), tseg->getEnd(), tseg->getStride(),
            icount,
            loop_body
         );
         break;
      }
#endif

    case _ListSeg_: {
      const ListSegment* tseg = static_cast<const ListSegment*>(iseg);
      impl::forall_Icount(SEG_EXEC_POLICY_T(), *tseg, icount, loop_body);
      break;
    }

    default: {
    }

  }  // switch on segment type
}

/*!
 ******************************************************************************
 *
 * \brief  Generic wrapper for IndexSet policies to allow the use of normal
 * policies with them.
 *
 ******************************************************************************
 */
// TODO: this should be with the IndexSet class, really it should be part of
// its built-in iterator, but we need to address the include snarl first
template <typename SEG_EXEC_POLICY_T, typename LOOP_BODY>
struct rangeListExecutor {
  constexpr rangeListExecutor(LOOP_BODY&& body) : body(body) {}
  RAJA_INLINE
  void operator()(const IndexSetSegInfo& seg_info)
  {
    executeRangeList_forall<SEG_EXEC_POLICY_T>(&seg_info, body);
  }

private:
  // LOOP_BODY body;
  typename std::remove_reference<LOOP_BODY>::type body;
};

template <typename SEG_EXEC_POLICY_T, typename LOOP_BODY>
constexpr RAJA_INLINE rangeListExecutor<SEG_EXEC_POLICY_T, LOOP_BODY>
makeRangeListExecutor(LOOP_BODY&& body)
{
  return rangeListExecutor<SEG_EXEC_POLICY_T, LOOP_BODY>(body);
}

template <typename SEG_IT_POLICY_T,
          typename SEG_EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE void forall(
    IndexSet::ExecPolicy<SEG_IT_POLICY_T, SEG_EXEC_POLICY_T>,
    const IndexSet& iset,
    LOOP_BODY loop_body)
{
  impl::forall(SEG_IT_POLICY_T(),
               iset,
               makeRangeListExecutor<SEG_EXEC_POLICY_T>(loop_body));
}

template <typename SEG_EXEC_POLICY_T, typename LOOP_BODY>
struct rangeListIcountExecutor {
  constexpr rangeListIcountExecutor(LOOP_BODY&& body) : body(body) {}
  RAJA_INLINE
  void operator()(const IndexSetSegInfo& seg_info)
  {
    executeRangeList_forall_Icount<SEG_EXEC_POLICY_T>(&seg_info, body);
  }

private:
  typename std::remove_reference<LOOP_BODY>::type body;
  // LOOP_BODY body;
};

template <typename SEG_EXEC_POLICY_T, typename LOOP_BODY>
constexpr RAJA_INLINE rangeListIcountExecutor<SEG_EXEC_POLICY_T, LOOP_BODY>
makeRangeListIcountExecutor(LOOP_BODY&& body)
{
  return rangeListIcountExecutor<SEG_EXEC_POLICY_T, LOOP_BODY>(body);
}

template <typename SEG_IT_POLICY_T,
          typename SEG_EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE void forall_Icount(
    IndexSet::ExecPolicy<SEG_IT_POLICY_T, SEG_EXEC_POLICY_T>,
    const IndexSet& iset,
    LOOP_BODY loop_body)
{
  // no need for icount variant here
  impl::forall(SEG_IT_POLICY_T(),
               iset,
               makeRangeListIcountExecutor<SEG_EXEC_POLICY_T>(loop_body));
}

}  // end of namespace impl
}  // end of namespace RAJA

#endif  // RAJA_rangelist_forall_HPP
