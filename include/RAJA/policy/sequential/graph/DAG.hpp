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
  DAGExec() = default;

  DAGExec(DAGExec const&) = default;
  DAGExec(DAGExec&&) = default;

  DAGExec& operator=(DAGExec const&) = default;
  DAGExec& operator=(DAGExec&&) = default;


  bool empty() const
  {
    return !m_node_data;
  }

  resources::EventProxy<GraphResource> exec(GraphResource& gr)
  {
    if (!empty()) {
      gr.wait();
      for (detail::NodeExec& ne : m_node_execs) {
        ne.exec(/*gr*/);
      }
    }
    return resources::EventProxy<GraphResource>(&gr);
  }

  resources::EventProxy<GraphResource> exec()
  {
    auto& gr = GraphResource::get_default();
    return exec(gr);
  }

private:
  friend DAG;

  using node_data_container = typename DAG::node_data_container;

  std::vector<detail::NodeExec> m_node_execs;
  std::shared_ptr<node_data_container> m_node_data;

  DAGExec(DAG& dag)
    : m_node_data(dag.m_node_data)
  {
    m_node_execs.reserve(dag.m_node_connections.size());
    // populate m_node_execs in a correct order
    dag.forward_traverse(
          [](detail::NodeConnections&) {
            // do nothing
          },
          [&](detail::NodeConnections& node) {
            node_data_container& container = *m_node_data;
            detail::NodeData* data = container[node.get_node_id()].get();
            m_node_execs.emplace_back(data);
          },
          [](detail::NodeConnections&) {
            // do nothing
          });
  }
};

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif
