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

#if defined(RAJA_ENABLE_OPENMP_TASK_DEPEND)

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

template < typename GraphResource >
struct DAGExec<omp_task_graph, GraphResource>
{
private:
  using base_node_type = typename DAG<omp_task_graph, GraphResource>::base_node_type;

public:
  resources::EventProxy<GraphResource> operator()(
      DAG<omp_task_graph, GraphResource>& dag, GraphResource& gr)
  {
#pragma omp parallel
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

#pragma omp task
      {
        node->exec(gr);
        for (base_node_type* child : node->m_children) {
          traverse(child, gr);
        }
      } // end omp task
    }
  }
};

}  // namespace detail

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_OPENMP guard

#endif  // closing endif for header file include guard
