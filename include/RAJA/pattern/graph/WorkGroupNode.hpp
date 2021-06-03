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

#ifndef RAJA_pattern_graph_WorkGroupNode_HPP
#define RAJA_pattern_graph_WorkGroupNode_HPP

#include "RAJA/config.hpp"

#include <utility>
#include <type_traits>

#include "RAJA/pattern/WorkGroup.hpp"

#include "RAJA/pattern/graph/DAG.hpp"
#include "RAJA/pattern/graph/Node.hpp"

#include "RAJA/util/camp_aliases.hpp"

namespace RAJA
{

namespace expt
{

namespace graph
{

template < typename WORKGROUP_POLICY_T,
           typename INDEX_T,
           typename EXTRA_ARGS_T,
           typename ALLOCATOR_T>
struct WorkGroupNode;

namespace detail
{

template < typename WORKGROUP_POLICY_T,
           typename INDEX_T,
           typename EXTRA_ARGS_T,
           typename ALLOCATOR_T>
struct WorkGroupArgs;

template < typename EXEC_POLICY_T,
           typename ORDER_POLICY_T,
           typename STORAGE_POLICY_T,
           typename INDEX_T,
           typename ... Args,
           typename ALLOCATOR_T>
struct WorkGroupArgs<WorkGroupPolicy<EXEC_POLICY_T,
                                     ORDER_POLICY_T,
                                     STORAGE_POLICY_T>,
                     INDEX_T,
                     xargs<Args...>,
                     ALLOCATOR_T> : NodeArgs
{
  using exec_policy = EXEC_POLICY_T;
  using order_policy = ORDER_POLICY_T;
  using storage_policy = STORAGE_POLICY_T;
  using policy = WorkGroupPolicy<exec_policy, order_policy, storage_policy>;
  using index_type = INDEX_T;
  using xarg_type = xargs<Args...>;
  using Allocator = ALLOCATOR_T;

  using node_type = WorkGroupNode<policy, index_type, xarg_type, Allocator>;

  WorkGroupArgs(Allocator const& aloc)
    : m_aloc(aloc)
  {
  }

  Allocator m_aloc;
};

}  // namespace detail

// policy by template
template < typename WORKGROUP_POLICY_T,
           typename INDEX_T,
           typename EXTRA_ARGS_T,
           typename ALLOCATOR_T >
RAJA_INLINE concepts::enable_if_t<
      detail::WorkGroupArgs<WORKGROUP_POLICY_T,
                            INDEX_T,
                            EXTRA_ARGS_T,
                            ALLOCATOR_T>,
      type_traits::is_WorkGroup_policy<WORKGROUP_POLICY_T>>
WorkGroup(ALLOCATOR_T const& aloc)
{
  return detail::WorkGroupArgs<WORKGROUP_POLICY_T,
                               INDEX_T,
                               EXTRA_ARGS_T,
                               ALLOCATOR_T>(
      aloc);
}

template < typename EXEC_POLICY_T,
           typename ORDER_POLICY_T,
           typename STORAGE_POLICY_T,
           typename INDEX_T,
           typename ... Args,
           typename ALLOCATOR_T>
struct WorkGroupNode<WorkGroupPolicy<EXEC_POLICY_T,
                                     ORDER_POLICY_T,
                                     STORAGE_POLICY_T>,
                     INDEX_T,
                     xargs<Args...>,
                     ALLOCATOR_T> : detail::NodeData
{
  using exec_policy = EXEC_POLICY_T;
  using order_policy = ORDER_POLICY_T;
  using storage_policy = STORAGE_POLICY_T;
  using policy = WorkGroupPolicy<exec_policy, order_policy, storage_policy>;
  using index_type = INDEX_T;
  using xarg_type = xargs<Args...>;
  using Allocator = ALLOCATOR_T;

  using workpool_type = WorkPool< policy, index_type, xarg_type, Allocator>;
  using workgroup_type = typename workpool_type::workgroup_type;
  using worksite_type  = typename workpool_type::worksite_type;

  using resource = typename workpool_type::resource_type;
  using args_type = detail::WorkGroupArgs<policy, INDEX_T, xarg_type, ALLOCATOR_T>;

  WorkGroupNode(Allocator const& aloc)
    : m_pool(aloc)
    , m_group(m_pool.instantiate())
    , m_site(m_group.run(Args()...))
    , m_args(Args()...)
    , m_instantiated(true)
  {
  }

  WorkGroupNode() = delete;

  WorkGroupNode(WorkGroupNode const&) = delete;
  WorkGroupNode(WorkGroupNode&&) = delete;

  WorkGroupNode& operator=(WorkGroupNode const&) = delete;
  WorkGroupNode& operator=(WorkGroupNode&&) = delete;

  WorkGroupNode(args_type const& args)
    : m_pool(args.m_aloc)
    , m_group(m_pool.instantiate())
    , m_site(m_group.run(Args()...))
    , m_args(Args()...)
    , m_instantiated(true)
  {
  }
  WorkGroupNode(args_type&& args)
    : m_pool(std::move(args.m_aloc))
    , m_group(m_pool.instantiate())
    , m_site(m_group.run(Args()...))
    , m_args(Args()...)
    , m_instantiated(true)
  {
  }

  WorkGroupNode& operator=(args_type const& args)
  {
    m_pool = workpool_type(args.m_aloc);
    m_group = m_pool.instantiate();
    m_site = m_group.run(Args()...);
    m_args = camp::tuple<Args...>(Args()...);
    m_instantiated = true;
    return *this;
  }
  WorkGroupNode& operator=(args_type&& args)
  {
    m_pool = workpool_type(std::move(args.m_aloc));
    m_group = m_pool.instantiate();
    m_site = m_group.run(Args()...);
    m_args = camp::tuple<Args...>(Args()...);
    m_instantiated = true;
    return *this;
  }

  virtual ~WorkGroupNode() = default;

  size_t num_loops() const
  {
    return m_pool.num_loops();
  }

  size_t storage_bytes() const
  {
    return m_pool.storage_bytes();
  }

  void reserve(size_t num_loops, size_t storage_bytes)
  {
    m_pool.reserve(num_loops, storage_bytes);
  }

  template < typename... EnqueueArgs >
  inline void enqueue(EnqueueArgs&&... args)
  {
    m_instantiated = false;
    m_pool.enqueue(std::forward<EnqueueArgs>(args)...);
  }

  void instantiate()
  {
    if (!m_instantiated) {
      m_instantiated = true;
      m_group = m_pool.instantiate();
    }
  }

  void set_args(Args... args)
  {
    m_args = camp::tuple<Args...>(std::forward<Args>(args)...);
  }

  void clear()
  {
    m_instantiated = true;
    m_site.clear();
    m_group.clear();
    m_pool.clear();
    m_args = camp::tuple<Args...>(Args()...);
    m_group = m_pool.instantiate();
    m_site = m_group.run(Args()...);
  }

protected:
  void exec() override
  {
    instantiate();
    return exec_impl();
  }

private:
  workpool_type m_pool;
  workgroup_type m_group;
  worksite_type m_site;
  camp::tuple<Args...> m_args;
  bool m_instantiated;

  template < camp::idx_t ... Is >
  void exec_impl_helper(resource& er, camp::idx_seq<Is...>)
  {
    m_site = m_group.run(er, RAJA::get<Is>(m_args)...);
  }

  void exec_impl()
  {
    resource& er = resource::get_default();

    exec_impl_helper(er, camp::make_idx_seq_t<sizeof...(Args)>());

    er.wait();
  }
};

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif
