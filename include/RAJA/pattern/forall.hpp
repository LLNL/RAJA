/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index set and segment iteration
 *          template methods that take an execution policy as a template
 *          parameter.
 *
 *          The templates for segments support the following usage pattern:
 *
 *             forall<exec_policy>( index set, loop body );
 *
 *          which is equivalent to:
 *
 *             forall( exec_policy(), index set, loop body );
 *
 *          The former is slightly more concise. Here, the execution policy
 *          type is associated with a tag struct defined in the exec_poilicy
 *          hearder file. Usage of the forall_Icount() is similar.
 *
 *          The forall() and forall_Icount() methods that take an index set
 *          take an execution policy of the form:
 *
 *          IndexSet::ExecPolicy< seg_it_policy, seg_exec_policy >
 *
 *          Here, the first template parameter determines the scheme for
 *          iteratiing over the index set segments and the second determines
 *          how each segment is executed.
 *
 *          The forall() templates accept a loop body argument that takes
 *          a single Index_type argument identifying the index of a loop
 *          iteration. The forall_Icount() templates accept a loop body that
 *          takes two Index_type arguments. The first is the number of the
 *          iteration in the indes set or segment, the second if the actual
 *          index of the loop iteration.
 *
 *
 *          IMPORTANT: Use of any of these methods requires a specialization
 *                     for the given index set type and execution policy.
 *
 ******************************************************************************
 */

#ifndef RAJA_forall_generic_HPP
#define RAJA_forall_generic_HPP

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

#include "RAJA/internal/Iterators.hpp"
#include "RAJA/policy/PolicyBase.hpp"

#include "RAJA/index/IndexSet.hpp"
#include "RAJA/index/ListSegment.hpp"
#include "RAJA/index/RangeSegment.hpp"

#include "RAJA/internal/fault_tolerance.hpp"
#include "RAJA/util/types.hpp"

#include <functional>
#include <iterator>
#include <type_traits>

#include "RAJA/internal/rangelist_forall.hpp"

namespace RAJA
{

//
//////////////////////////////////////////////////////////////////////
//
// Iteration over generic iterators
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over containers with icount
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T, typename LOOP_BODY>
RAJA_INLINE void forall_Icount(const IndexSet& c, LOOP_BODY loop_body)
{

  impl::forall_Icount(EXEC_POLICY_T(), c, loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over containers with icount
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T, typename Container, typename LOOP_BODY>
RAJA_INLINE void forall_Icount(Container&& c,
                               Index_type icount,
                               LOOP_BODY loop_body)
{
  using Iterator = decltype(std::begin(c));
  using category = typename std::iterator_traits<Iterator>::iterator_category;
  static_assert(
      std::is_base_of<std::random_access_iterator_tag, category>::value,
      "Iterators passed to RAJA must be Random Access or Contiguous iterators");

  impl::forall_Icount(EXEC_POLICY_T(),
                      std::forward<Container>(c),
                      icount,
                      loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over containers with a value-based policy
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T, typename Container, typename LOOP_BODY>
RAJA_INLINE void forall(EXEC_POLICY_T&& p, Container&& c, LOOP_BODY loop_body)
{
  using category =
      typename std::iterator_traits<decltype(std::begin(c))>::iterator_category;
  static_assert(
      std::is_base_of<std::random_access_iterator_tag, category>::value,
      "Iterators passed to RAJA must be Random Access or Contiguous iterators");

  // printf("running container\n");

  impl::forall(std::forward<EXEC_POLICY_T>(p),
               std::forward<Container>(c),
               loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over containers
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T, typename Container, typename LOOP_BODY>
RAJA_INLINE void forall(Container&& c, LOOP_BODY loop_body)
{
  using category =
      typename std::iterator_traits<decltype(std::begin(c))>::iterator_category;
  static_assert(
      std::is_base_of<std::random_access_iterator_tag, category>::value,
      "Iterators passed to RAJA must be Random Access or Contiguous iterators");

  // printf("running container\n");

  impl::forall(EXEC_POLICY_T(), std::forward<Container>(c), loop_body);
}

//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over index ranges.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief Generic iteration over index range.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T, typename LOOP_BODY>
RAJA_INLINE void forall(Index_type begin, Index_type end, LOOP_BODY loop_body)
{
  forall<EXEC_POLICY_T>(RangeSegment(begin, end), loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief Generic iteration over index range with index count.
 *
 *        NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T, typename LOOP_BODY>
RAJA_INLINE void forall_Icount(Index_type begin,
                               Index_type end,
                               Index_type icount,
                               LOOP_BODY loop_body)
{
  impl::forall_Icount(EXEC_POLICY_T(),
                      RangeSegment(begin, end),
                      icount,
                      loop_body);
}

//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over index ranges with stride.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief Generic iteration over index range with stride.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T, typename LOOP_BODY>
RAJA_INLINE void forall(Index_type begin,
                        Index_type end,
                        Index_type stride,
                        LOOP_BODY loop_body)
{
  impl::forall(EXEC_POLICY_T(),
               RangeStrideSegment(begin, end, stride),
               loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief Generic iteration over index range with stride with index count.
 *
 *        NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T, typename LOOP_BODY>
RAJA_INLINE void forall_Icount(Index_type begin,
                               Index_type end,
                               Index_type stride,
                               Index_type icount,
                               LOOP_BODY loop_body)
{
  impl::forall_Icount(EXEC_POLICY_T(),
                      RangeStrideSegment(begin, end, stride),
                      icount,
                      loop_body);
}

//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over indirection arrays.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Generic iteration over indices in indirection array.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T, typename LOOP_BODY>
RAJA_INLINE void forall(const Index_type* idx,
                        Index_type len,
                        LOOP_BODY loop_body)
{
  // turn into an iterator
  forall<EXEC_POLICY_T>(ListSegment(idx, len, Unowned), loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief  Generic iteration over indices in indirection array with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T, typename LOOP_BODY>
RAJA_INLINE void forall_Icount(const Index_type* idx,
                               Index_type len,
                               Index_type icount,
                               LOOP_BODY loop_body)
{
  // turn into an iterator
  forall_Icount<EXEC_POLICY_T>(ListSegment(idx, len, Unowned),
                               icount,
                               loop_body);
}


}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
