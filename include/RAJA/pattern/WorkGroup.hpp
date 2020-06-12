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

#include "RAJA/pattern/detail/WorkGroup.hpp"

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
          typename ... Args,
          typename ALLOCATOR_T>
struct WorkPool<WorkGroupPolicy<EXEC_POLICY_T,
                                ORDER_POLICY_T,
                                STORAGE_POLICY_T>,
                INDEX_T,
                xargs<Args...>,
                ALLOCATOR_T>
{
  using exec_policy = EXEC_POLICY_T;
  using order_policy = ORDER_POLICY_T;
  using storage_policy = STORAGE_POLICY_T;
  using policy = WorkGroupPolicy<exec_policy, order_policy, storage_policy>;
  using index_type = INDEX_T;
  using xarg_type = xargs<Args...>;
  using Allocator = ALLOCATOR_T;

  WorkPool(Allocator aloc)
    : m_storage(std::forward<Allocator>(aloc))
  { }

  WorkPool(WorkPool const&) = delete;
  WorkPool& operator=(WorkPool const&) = delete;

  WorkPool(WorkPool&&) = default;
  WorkPool& operator=(WorkPool&&) = default;

  inline WorkGroup<policy, index_type, xarg_type, Allocator> instantiate();

private:
  using storage_type = detail::WorkStorage<storage_policy,
                                           Allocator,
                                           index_type, Args...>;
  storage_type m_storage;
};

template <typename EXEC_POLICY_T,
          typename ORDER_POLICY_T,
          typename STORAGE_POLICY_T,
          typename INDEX_T,
          typename ... Args,
          typename ALLOCATOR_T>
struct WorkGroup<WorkGroupPolicy<EXEC_POLICY_T,
                                 ORDER_POLICY_T,
                                 STORAGE_POLICY_T>,
                 INDEX_T,
                 xargs<Args...>,
                 ALLOCATOR_T>
{
  using exec_policy = EXEC_POLICY_T;
  using order_policy = ORDER_POLICY_T;
  using storage_policy = STORAGE_POLICY_T;
  using policy = WorkGroupPolicy<exec_policy, order_policy, storage_policy>;
  using index_type = INDEX_T;
  using xarg_type = xargs<Args...>;
  using Allocator = ALLOCATOR_T;

  WorkGroup(WorkGroup const&) = delete;
  WorkGroup& operator=(WorkGroup const&) = delete;

  WorkGroup(WorkGroup&&) = default;
  WorkGroup& operator=(WorkGroup&&) = default;

  inline WorkSite<policy, index_type, xarg_type, Allocator> run(Args...);

private:
  friend WorkPool<WorkGroupPolicy<EXEC_POLICY_T,
                                  ORDER_POLICY_T,
                                  STORAGE_POLICY_T>,
                   INDEX_T,
                   xargs<Args...>,
                   ALLOCATOR_T>;

  using storage_type = detail::WorkStorage<storage_policy,
                                           Allocator,
                                           index_type, Args...>;

  storage_type m_storage;

  WorkGroup(storage_type&& storage)
    : m_storage(std::move(storage))
  { }
};

template <typename EXEC_POLICY_T,
          typename ORDER_POLICY_T,
          typename STORAGE_POLICY_T,
          typename INDEX_T,
          typename ... Args,
          typename ALLOCATOR_T>
struct WorkSite<WorkGroupPolicy<EXEC_POLICY_T,
                                ORDER_POLICY_T,
                                STORAGE_POLICY_T>,
                INDEX_T,
                xargs<Args...>,
                ALLOCATOR_T>
{
  using exec_policy = EXEC_POLICY_T;
  using order_policy = ORDER_POLICY_T;
  using storage_policy = STORAGE_POLICY_T;
  using policy = WorkGroupPolicy<exec_policy, order_policy, storage_policy>;
  using index_type = INDEX_T;
  using xarg_type = xargs<Args...>;
  using Allocator = ALLOCATOR_T;

  WorkSite(WorkSite const&) = delete;
  WorkSite& operator=(WorkSite const&) = delete;

  WorkSite(WorkSite&&) = default;
  WorkSite& operator=(WorkSite&&) = default;

private:
  friend WorkGroup<WorkGroupPolicy<EXEC_POLICY_T,
                                   ORDER_POLICY_T,
                                   STORAGE_POLICY_T>,
                    INDEX_T,
                    xargs<Args...>,
                    ALLOCATOR_T>;

  WorkSite()
  { }
};


template <typename EXEC_POLICY_T,
          typename ORDER_POLICY_T,
          typename STORAGE_POLICY_T,
          typename INDEX_T,
          typename ... Args,
          typename ALLOCATOR_T>
inline
WorkGroup<WorkGroupPolicy<EXEC_POLICY_T,
                          ORDER_POLICY_T,
                          STORAGE_POLICY_T>,
          INDEX_T,
          xargs<Args...>,
          ALLOCATOR_T>
WorkPool<WorkGroupPolicy<EXEC_POLICY_T,
                         ORDER_POLICY_T,
                         STORAGE_POLICY_T>,
         INDEX_T,
         xargs<Args...>,
         ALLOCATOR_T>::instantiate()
{
  return WorkGroup<WorkGroupPolicy<EXEC_POLICY_T,
                                   ORDER_POLICY_T,
                                   STORAGE_POLICY_T>,
                   INDEX_T,
                   xargs<Args...>,
                   ALLOCATOR_T>{std::move(m_storage)};
}

template <typename EXEC_POLICY_T,
          typename ORDER_POLICY_T,
          typename STORAGE_POLICY_T,
          typename INDEX_T,
          typename ... Args,
          typename ALLOCATOR_T>
inline
WorkSite<WorkGroupPolicy<EXEC_POLICY_T,
                         ORDER_POLICY_T,
                         STORAGE_POLICY_T>,
         INDEX_T,
         xargs<Args...>,
         ALLOCATOR_T>
WorkGroup<WorkGroupPolicy<EXEC_POLICY_T,
                          ORDER_POLICY_T,
                          STORAGE_POLICY_T>,
          INDEX_T,
          xargs<Args...>,
          ALLOCATOR_T>::run(Args... args)
{
  RAJA_UNUSED_VAR(std::forward<Args>(args)...);
  return WorkSite<WorkGroupPolicy<EXEC_POLICY_T,
                                   ORDER_POLICY_T,
                                   STORAGE_POLICY_T>,
                   INDEX_T,
                   xargs<Args...>,
                   ALLOCATOR_T>{};
}

}  // namespace RAJA

#endif  // closing endif for header file include guard
