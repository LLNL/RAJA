/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file providing RAJA WorkPool and WorkGroup declarations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_PATTERN_WorkGroup_HPP
#define RAJA_PATTERN_WorkGroup_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/Operators.hpp"
#include "RAJA/util/macros.hpp"

#include "RAJA/policy/WorkGroup.hpp"

namespace RAJA
{

/*!
 ******************************************************************************
 *
 * \brief  xargs alias.
 *
 * Usage example:
 *
 * \verbatim

   WorkPool<WorkGroup_policy, Index_type, xargs<int*, int>, Allocator> pool(allocator);

   pool.enqueue(..., [=] (Index_type i, int* xarg0, int xarg1) {
      xarg0[i] = xarg1;
   });

   WorkGroup<WorkGroup_policy, Index_type, xargs<int*, int>, Allocator> group = pool.instantiate();

   int* xarg0 = ...;
   int xarg1 = ...;
   WorkSite<WorkGroup_policy, Index_type, xargs<int*, int>, Allocator> site = group.run(xarg0, xarg1);

 * \endverbatim
 *
 ******************************************************************************
 */
template < typename ... Args >
using xargs = camp::list<Args...>;

//
// Forward declarations for WorkPool and WorkGroup templates.
// Actual classes appear in forall_*.hxx header files.
//

/*!
 ******************************************************************************
 *
 * \brief  WorkPool class template.
 *
 * Usage example:
 *
 * \verbatim

   WorkPool<WorkGroup_policy, Index_type, xargs<>, Allocator> pool(allocator);

   Real_ptr data = ...;

   pool.enqueue( ..., [=] (Index_type i) {
      data[i] = 1;
   });

   WorkGroup<WorkGroup_policy, Index_type, xargs<>, Allocator> group = pool.instantiate();

 * \endverbatim
 *
 ******************************************************************************
 */
template <typename WORKGROUP_POLICY_T,
          typename INDEX_T,
          typename EXTRA_ARGS_T,
          typename ALLOCATOR_T>
struct WorkPool;

/*!
 ******************************************************************************
 *
 * \brief  WorkGroup class template. Owns loops from an instantiated WorkPool.
 *
 * Usage example:
 *
 * \verbatim

   WorkGroup<WorkGroup_policy, Index_type, xargs<>, Allocator> group = pool.instantiate();

   WorkSite<WorkGroup_policy, Index_type, xargs<>, Allocator> site = group.run();

 * \endverbatim
 *
 ******************************************************************************
 */
template <typename WORKGROUP_POLICY_T,
          typename INDEX_T,
          typename EXTRA_ARGS_T,
          typename ALLOCATOR_T>
struct WorkGroup;

/*!
 ******************************************************************************
 *
 * \brief  WorkSite class template. Owns per run objects from a single run of a WorkGroup.
 *
 * Usage example:
 *
 * \verbatim

   WorkSite<WorkGroup_policy, Index_type, xargs<>, Allocator> site = group.run();

   site.synchronize();

 * \endverbatim
 *
 ******************************************************************************
 */
template <typename WORKGROUP_POLICY_T,
          typename INDEX_T,
          typename EXTRA_ARGS_T,
          typename ALLOCATOR_T>
struct WorkSite;


template <typename EXEC_POLICY_T,
          typename ORDER_POLICY_T,
          typename STORAGE_POLICY_T,
          typename INDEX_T,
          typename ... Xargs,
          typename ALLOCATOR_T>
struct WorkPool<WorkGroupPolicy<EXEC_POLICY_T,
                                ORDER_POLICY_T,
                                STORAGE_POLICY_T>,
                INDEX_T,
                xargs<Xargs...>,
                ALLOCATOR_T>
{
  using exec_policy = EXEC_POLICY_T;
  using order_policy = ORDER_POLICY_T;
  using storage_policy = STORAGE_POLICY_T;
  using policy = WorkGroupPolicy<exec_policy, order_policy, storage_policy>;
  using index_type = INDEX_T;
  using xarg_type = xargs<Xargs...>;
  using Allocator = ALLOCATOR_T;

  WorkPool(Allocator aloc)
    : m_aloc(std::forward<Allocator>(aloc))
  { }

private:
  Allocator m_aloc;
};

}  // namespace RAJA

#endif  // closing endif for header file include guard
