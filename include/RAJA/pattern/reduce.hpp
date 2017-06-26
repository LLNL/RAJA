/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file providing RAJA reduction declarations.
 *
 ******************************************************************************
 */

#ifndef RAJA_reducers_HPP
#define RAJA_reducers_HPP

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

///
/// Define max number of reductions allowed within a RAJA traversal
/// (sizes of shared memory blocks for reductions are set based on this value)
///
#define RAJA_MAX_REDUCE_VARS (8)

namespace RAJA
{

///
/// Macros for type agnostic reduction operations.
///
#define RAJA_MIN(a, b) (((b) < (a)) ? (b) : (a))
///
#define RAJA_MAX(a, b) (((b) > (a)) ? (b) : (a))

///
/// Macros to support structs used in minmaxloc operations
#define RAJA_MINLOC(a, b) (((b.val) < (a.val)) ? (b) : (a))
///
#define RAJA_MAXLOC(a, b) (((b.val) > (a.val)) ? (b) : (a))

///
/// Macros to support unstructured minmaxloc operations
#define RAJA_MINLOC_UNSTRUCTURED(set_val, set_idx, a_val, a_idx, b_val, b_idx) \
  set_idx = ((b_val) < (a_val) ? (b_idx) : (a_idx));                           \
  set_val = ((b_val) < (a_val) ? (b_val) : (a_val));
///
#define RAJA_MAXLOC_UNSTRUCTURED(set_val, set_idx, a_val, a_idx, b_val, b_idx) \
  set_idx = ((b_val) > (a_val) ? (b_idx) : (a_idx));                           \
  set_val = ((b_val) > (a_val) ? (b_val) : (a_val));

//
// Forward declarations for reduction templates.
// Actual classes appear in forall_*.hxx header files.
//
// IMPORTANT: reduction policy parameter must be consistent with loop
//            execution policy type.
//
// Also, mutliple reductions using different reduction operations may be
// combined in a single RAJA forall() construct.
//

/*!
 ******************************************************************************
 *
 * \brief  Min reducer class template.
 *
 * Usage example:
 *
 * \verbatim

   Real_ptr data = ...;
   ReduceMin<reduce_policy, Real_type> my_min(init_val);

   forall<exec_policy>( ..., [=] (Index_type i) {
      my_min.min(data[i]);
   }

   Real_type minval = my_min;

 * \endverbatim
 *
 ******************************************************************************
 */
template <typename REDUCE_POLICY_T, typename T>
class ReduceMin;

/*!
 ******************************************************************************
 *
 * \brief  Min-loc reducer class template.
 *
 * Usage example:
 *
 * \verbatim

   Real_ptr data = ...;
   ReduceMin<reduce_policy, Real_type> my_min(init_val, -1);

   forall<exec_policy>( ..., [=] (Index_type i) {
      my_min.minloc(data[i], i);
   }

   Real_type minval = my_min;
   Index_type minloc = my_min.getMinLoc();

 * \endverbatim
 *
 ******************************************************************************
 */
template <typename REDUCE_POLICY_T, typename T>
class ReduceMinLoc;

/*!
 ******************************************************************************
 *
 * \brief  Max reducer class template.
 *
 * Usage example:
 *
 * \verbatim

   Real_ptr data = ...;
   ReduceMax<reduce_policy, Real_type> my_max(init_val);

   forall<exec_policy>( ..., [=] (Index_type i) {
      my_max.max(data[i]);
   }

   Real_type maxval = my_max;

 * \endverbatim
 *
 ******************************************************************************
 */
template <typename REDUCE_POLICY_T, typename T>
class ReduceMax;

/*!
 ******************************************************************************
 *
 * \brief  Max-loc reducer class template.
 *
 * Usage example:
 *
 * \verbatim

   Real_ptr data = ...;
   ReduceMax<reduce_policy, Real_type> my_max(init_val, -1);

   forall<exec_policy>( ..., [=] (Index_type i) {
      my_max.maxloc(data[i], i);
   }

   Real_type maxval = my_max;
   Index_type maxloc = my_max.getMaxLoc();

 * \endverbatim
 *
 ******************************************************************************
 */
template <typename REDUCE_POLICY_T, typename T>
class ReduceMaxLoc;

/*!
 ******************************************************************************
 *
 * \brief  Sum reducer class template.
 *
 * Usage example:
 *
 * \verbatim

   Real_ptr data = ...;
   ReduceSum<reduce_policy, Real_type> my_sum(init_val);

   forall<exec_policy>( ..., [=] (Index_type i) {
      my_sum += data[i];
   }

   Real_type sum = my_sum;

 * \endverbatim
 *
 ******************************************************************************
 */
template <typename REDUCE_POLICY_T, typename T>
class ReduceSum;

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
