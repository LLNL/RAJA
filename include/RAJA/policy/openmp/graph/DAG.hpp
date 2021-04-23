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
  resources::EventProxy<GraphResource> operator()(
      DAG<omp_task_graph, GraphResource>& dag, GraphResource& gr)
  {
#pragma omp parallel
#pragma omp single nowait
    {
      dag.forward_traverse(
          [](Node<GraphResource>*) {
            // do nothing on examine
          },
          [&](Node<GraphResource>* node) {
            // exec on enter

            // nodes express in dependencies through themselves and
            // express out dependencies through their child nodes
            size_t num_children = node->m_children.size();
            Node<GraphResource>** children = &node->m_children[0];

#pragma omp task depend(in:node[0:1]) depend(out:children[0:num_children][0:1])
            {
              node->exec(gr);
            }

          },
          [](Node<GraphResource>*) {
            // do nothing on exit
          });
    } // end omp parallel
    return resources::EventProxy<GraphResource>(&gr);
  }
};

}  // namespace detail

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_OPENMP guard

#endif  // closing endif for header file include guard
