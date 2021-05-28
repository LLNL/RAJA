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

#if defined(RAJA_ENABLE_OPENMP_TASK) && defined(RAJA_ENABLE_OPENMP_ATOMIC_CAPTURE)

template < typename GraphResource >
struct DAGExec<omp_task_atomic_graph, GraphResource>
    : detail::DAGExecBase<omp_task_atomic_graph, GraphResource>
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
#pragma omp parallel default(none) shared(gr)
#pragma omp single nowait
      {
        for (NodeExecConnections* connections : m_first_node_execs) {
          exec_traverse(connections, gr);
        }
      } // end omp parallel
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

  struct NodeExecConnections : detail::NodeExec
  {
    NodeExecConnections(detail::NodeConnections& connections, node_data_container& container)
      : detail::NodeExec(container[connections.get_node_id()].get())
      , m_parent_count(connections.m_parent_count)
      , m_count(connections.m_count)
    {
      m_children.reserve(connections.m_children.size());
    }

    ~NodeExecConnections() = default;

    void add_children(detail::NodeConnections& connections, NodeExecConnections* node_connections)
    {
      for (size_t child_id : connections.m_children) {
        NodeExecConnections* exec_child = &node_connections[child_id];
        m_children.emplace_back(exec_child);
      }
    }

    int m_parent_count;
    int m_count;
    std::vector<NodeExecConnections*> m_children;
  };

  static void exec_traverse(NodeExecConnections* connections, GraphResource& gr)
  {
#pragma omp task default(none) firstprivate(connections) shared(gr)
    {
      connections->exec(/*gr*/);
      for (NodeExecConnections* child : connections->m_children) {
        int node_count;
#pragma omp atomic capture
        node_count = ++child->m_count;
        if (node_count == child->m_parent_count) {
          child->m_count = 0;
          exec_traverse(child, gr);
        }
      }
    } // end omp task
  }

  std::vector<NodeExecConnections*> m_first_node_execs;
  std::vector<NodeExecConnections> m_node_execs;
  std::shared_ptr<node_data_container> m_node_data;

  DAGExec(DAG& dag)
    : m_node_data(dag.m_node_data)
  {
    // make NodeExecConnections from NodeConnections
    m_node_execs.reserve(dag.m_node_connections.size());
    for (detail::NodeConnections& connections : dag.m_node_connections) {
      m_node_execs.emplace_back(connections, *m_node_data);
    }
    // add children in second pass as child pointers point into same array
    for (size_t i = 0; i < dag.m_node_connections.size(); ++i) {
      detail::NodeConnections& connections = dag.m_node_connections[i];
      m_node_execs[i].add_children(connections, m_node_execs.data());
      if (m_node_execs[i].m_parent_count == 0) {
        m_first_node_execs.emplace_back(&m_node_execs[i]);
      }
    }
  }
};

#endif  // closing endif for RAJA_ENABLE_OPENMP_TASK && RAJA_ENABLE_OPENMP_ATOMIC_CAPTURE guard

#if defined(RAJA_ENABLE_OPENMP_TASK_DEPEND) && defined(RAJA_ENABLE_OPENMP_ITERATOR)

template < typename GraphResource >
struct DAGExec<omp_task_depend_graph, GraphResource>
    : detail::DAGExecBase<omp_task_depend_graph, GraphResource>
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
#pragma omp parallel default(none) shared(gr)
#pragma omp single nowait
      {
        for (NodeExecConnections& connections : m_node_execs) {
          detail::NodeExec* node_exec = &connections;
          detail::NodeData* node_data = node_exec->m_nodeData;

          size_t num_children = connections->m_children.size();
          detail::NodeData** child_data = connections->m_children.data();

#pragma omp task default(none) firstprivate(node_exec) /*shared(gr)*/ \
                 depend(in:node_data[0:1]) \
                 depend(iterator(size_t it = 0:num_children), out:child_data[it][0:1])
          {
            node_exec->exec(/*gr*/);
          } // end omp task
        }
      } // end omp parallel
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
  using node_data_container_values = typename node_data_container::value_type;

  struct NodeExecConnections : detail::NodeExec
  {
    NodeExecConnections(detail::NodeConnections& connections, node_data_container& container)
      : detail::NodeExec(container[connections.get_node_id()].get())
    {
      for (size_t child_id : connections.m_children) {
        m_children.emplace_back(container[child_id].get());
      }
    }

    ~NodeExecConnections() = default;

    std::vector<detail::NodeData*> m_children;
  };

  std::vector<NodeExecConnections> m_node_execs;
  std::shared_ptr<node_data_container> m_node_data;

  DAGExec(DAG& dag)
    : m_node_data(dag.m_node_data)
  {
    // populate m_node_execs in a correct order
    dag.forward_traverse(
          [](detail::NodeConnections&) {
            // do nothing
          },
          [&](detail::NodeConnections& node_connections) {
            node_data_container& container = *m_node_data;
            m_node_execs.emplace_back(node_connections, m_node_data);
          },
          [](detail::NodeConnections&) {
            // do nothing
          });
  }
};

#endif  // closing endif for RAJA_ENABLE_OPENMP_TASK_DEPEND && RAJA_ENABLE_OPENMP_ITERATOR guard

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_OPENMP guard

#endif  // closing endif for header file include guard
