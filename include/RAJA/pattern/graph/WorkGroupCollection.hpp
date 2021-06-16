/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing the core components of RAJA::graph::Node
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_graph_WorkGroupCollection_HPP
#define RAJA_pattern_graph_WorkGroupCollection_HPP

#include "RAJA/config.hpp"

#include <utility>
#include <type_traits>
#include <vector>

#include "RAJA/pattern/WorkGroup/WorkStorage.hpp"
#include "RAJA/pattern/WorkGroup/WorkRunner.hpp"

#include "RAJA/pattern/graph/Collection.hpp"

namespace RAJA
{

namespace expt
{

namespace graph
{

template < typename ExecutionPolicy, typename Container, typename LoopBody >
struct FusibleForallNode;

template < typename EXEC_POLICY_T,
           typename ORDER_POLICY_T,
           typename INDEX_T,
           typename EXTRA_ARGS_T,
           typename ALLOCATOR_T >
struct WorkGroupCollection;

namespace detail
{

template < typename EXEC_POLICY_T,
           typename ORDER_POLICY_T,
           typename INDEX_T,
           typename EXTRA_ARGS_T,
           typename ALLOCATOR_T >
struct WorkGroupCollectionArgs : ::RAJA::expt::graph::detail::CollectionArgs
{
  using collection_type = ::RAJA::expt::graph::WorkGroupCollection<
                                                   EXEC_POLICY_T,
                                                   ORDER_POLICY_T,
                                                   INDEX_T,
                                                   EXTRA_ARGS_T,
                                                   ALLOCATOR_T >;

  WorkGroupCollectionArgs(ALLOCATOR_T const& aloc)
    : m_aloc(aloc)
  {
  }

  ALLOCATOR_T m_aloc;
};

}  // namespace detail


template < typename EXEC_POLICY_T,
           typename ORDER_POLICY_T,
           typename INDEX_T,
           typename EXTRA_ARGS_T,
           typename ALLOCATOR_T >
::RAJA::expt::graph::detail::WorkGroupCollectionArgs<EXEC_POLICY_T,
                                                     ORDER_POLICY_T,
                                                     INDEX_T,
                                                     EXTRA_ARGS_T,
                                                     ALLOCATOR_T>
WorkGroup(ALLOCATOR_T const& aloc)
{
  return ::RAJA::expt::graph::detail::WorkGroupCollectionArgs<EXEC_POLICY_T,
                                                              ORDER_POLICY_T,
                                                              INDEX_T,
                                                              EXTRA_ARGS_T,
                                                              ALLOCATOR_T>(
      aloc);
}


template <typename EXEC_POLICY_T,
          typename ORDER_POLICY_T,
          typename INDEX_T,
          typename ... Args,
          typename ALLOCATOR_T>
struct WorkGroupCollection<EXEC_POLICY_T,
                           ORDER_POLICY_T,
                           INDEX_T,
                           ::RAJA::xargs<Args...>,
                           ALLOCATOR_T>
    : ::RAJA::expt::graph::detail::Collection
{
  using base = ::RAJA::expt::graph::detail::Collection;

  using exec_policy = EXEC_POLICY_T;
  using order_policy = ORDER_POLICY_T;
  using storage_policy = ::RAJA::array_of_pointers;
  using policy = WorkGroupPolicy<exec_policy, order_policy, storage_policy>;
  using index_type = INDEX_T;
  using xarg_type = xargs<Args...>;
  using Allocator = ALLOCATOR_T;
  using resource_type = typename ::RAJA::resources::get_resource<exec_policy>::type;

  using args_type = ::RAJA::expt::graph::detail::WorkGroupCollectionArgs<
                                                     exec_policy,
                                                     order_policy,
                                                     index_type,
                                                     xarg_type,
                                                     Allocator>;

  WorkGroupCollection() = delete;

  WorkGroupCollection(WorkGroupCollection const&) = delete;
  WorkGroupCollection(WorkGroupCollection&&) = delete;

  WorkGroupCollection& operator=(WorkGroupCollection const&) = delete;
  WorkGroupCollection& operator=(WorkGroupCollection&&) = delete;

  WorkGroupCollection(size_t id, args_type const& args)
    : base(id)
    , m_aloc(args.m_aloc)
  {
  }

  virtual ~WorkGroupCollection()
  {
    for (pointer_and_size& value_and_size_ptr : m_values) {
      storage_type::destroy_value(m_aloc, value_and_size_ptr);
    }
  }

  // make_FusedNode()

protected:
  template < typename, typename, typename >
  friend struct FusibleForallNode;

  using workrunner_type = ::RAJA::detail::WorkRunner<exec_policy,
                                                     order_policy,
                                                     Allocator,
                                                     index_type,
                                                     Args...>;

  // The policy indicating where the call function is invoked
  using vtable_exec_policy = typename workrunner_type::vtable_exec_policy;
  using vtable_type = typename workrunner_type::vtable_type;
  template < typename Container, typename LoopBody >
  using runner_holder_type = typename workrunner_type::template holder_type<Container, LoopBody>;
  template < typename Container, typename LoopBody >
  using runner_caller_type = typename workrunner_type::template caller_type<Container, LoopBody>;

  using storage_type = ::RAJA::detail::WorkStorage<storage_policy,
                                                   Allocator,
                                                   vtable_type>;

  using pointer_and_size = typename storage_type::pointer_and_size;
  using value_type = typename storage_type::value_type;



  std::vector<pointer_and_size> m_values;
  Allocator m_aloc;


  // Create items with storage for use in storage_type.
  // Note that this object owns the storage not instances of storage_type, so
  // where storage_type is created take care to avoid double freeing the items.
  template < typename Container, typename LoopBody >
  runner_holder_type<::camp::decay<Container>, ::camp::decay<LoopBody>>*
  emplace(Container&& c, LoopBody&& body)
  {
    using holder = runner_caller_type<::camp::decay<Container>, ::camp::decay<LoopBody>>;

    const vtable_type* vtable = ::RAJA::detail::get_Vtable<holder, vtable_type>(
        vtable_exec_policy{});

    pointer_and_size value = storage_type::template create_value<holder>(
        m_aloc, vtable, std::forward<Container>(c), std::forward<LoopBody>(body));
    m_values.emplace_back(std::move(value));

    m_num_nodes = m_values.size();

    return value_type::template get_holder<holder>(m_values.back().ptr);
  }
};

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif
