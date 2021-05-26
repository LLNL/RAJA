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

#ifndef RAJA_policy_sequential_graph_DAG_HPP
#define RAJA_policy_sequential_graph_DAG_HPP

#include "RAJA/config.hpp"

#include "RAJA/pattern/graph/DAG.hpp"
#include "RAJA/pattern/graph/Node.hpp"

namespace RAJA
{

namespace expt
{

namespace graph
{

template < typename GraphResource >
struct DAGExec<seq_graph, GraphResource>
    : detail::DAGExecBase<seq_graph, GraphResource>
{
  resources::EventProxy<GraphResource> exec(GraphResource& gr)
  {
    gr.wait();
    // exec all nodes in a correct order
    m_dag->forward_traverse(
          [](Node*) {
            // do nothing
          },
          [&](Node* node) {
            node->exec(/*gr*/);
          },
          [](Node*) {
            // do nothing
          });
    return resources::EventProxy<GraphResource>(&gr);
  }

  resources::EventProxy<GraphResource> exec()
  {
    auto& gr = GraphResource::get_default();
    return exec(gr);
  }

private:
  friend DAG;

  DAGExec(DAG* dag)
    : m_dag(dag)
  { }

  DAG* m_dag;
};

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif
