/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing the core components of RAJA::graph::DAG
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_openmp_graph_DAG_HPP
#define RAJA_policy_openmp_graph_DAG_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_OPENMP)

#include "RAJA/pattern/graph/DAG.hpp"
#include "RAJA/pattern/graph/Node.hpp"

namespace RAJA
{

namespace expt
{

namespace graph
{

namespace detail
{

#if defined(RAJA_ENABLE_OPENMP_TASK) && defined(RAJA_ENABLE_OPENMP_ATOMIC_CAPTURE)

template < typename GraphResource >
struct DAGExec<omp_task_atomic_graph, GraphResource>
{
private:
  using base_node_type = typename DAG<omp_task_atomic_graph, GraphResource>::base_node_type;

public:
  resources::EventProxy<GraphResource> operator()(
      DAG<omp_task_atomic_graph, GraphResource>& dag, GraphResource& gr)
  {
#pragma omp parallel default(none) shared(dag, gr)
#pragma omp single nowait
    {
      for (base_node_type* child : dag.m_children) {
        traverse(child, gr);
      }
    } // end omp parallel
    return resources::EventProxy<GraphResource>(&gr);
  }
private:

  static void traverse(base_node_type* node, GraphResource& gr)
  {
    int node_count;
#pragma omp atomic capture
    node_count = ++node->m_count;
    if (node_count == node->m_parent_count) {
      node->m_count = 0;

#pragma omp task default(none) firstprivate(node) shared(gr)
      {
        node->exec(gr);
        for (base_node_type* child : node->m_children) {
          traverse(child, gr);
        }
      } // end omp task
    }
  }
};

#endif  // closing endif for RAJA_ENABLE_OPENMP_TASK && RAJA_ENABLE_OPENMP_ATOMIC_CAPTURE guard

#if defined(RAJA_ENABLE_OPENMP_TASK_DEPEND) && defined(RAJA_ENABLE_OPENMP_ITERATOR)

template < typename GraphResource >
struct DAGExec<omp_task_depend_graph, GraphResource>
{
private:
  using base_node_type = typename DAG<omp_task_depend_graph, GraphResource>::base_node_type;

public:
  resources::EventProxy<GraphResource> operator()(
      DAG<omp_task_depend_graph, GraphResource>& dag, GraphResource& gr)
  {
#pragma omp parallel default(none) shared(dag, gr)
#pragma omp single nowait
    {
      // exec all nodes in a correct order
      dag.forward_traverse(
            [](Node<GraphResource>*) {
              // do nothing
            },
            [&](Node<GraphResource>* node) {
              size_t num_children = node->m_children.size();
              base_node_type** children = node->m_children.data();
#pragma omp task default(none) firstprivate(node) shared(gr) \
                 depend(in:node[0:1]) \
                 depend(iterator(size_t it = 0:num_children), out:children[it][0:1])
              {
                node->exec(gr);
              } // end omp task
            },
            [](Node<GraphResource>*) {
              // do nothing
            });
    } // end omp parallel
    return resources::EventProxy<GraphResource>(&gr);
  }
};

#endif  // closing endif for RAJA_ENABLE_OPENMP_TASK_DEPEND && RAJA_ENABLE_OPENMP_ITERATOR guard

}  // namespace detail

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_OPENMP guard

#endif  // closing endif for header file include guard
