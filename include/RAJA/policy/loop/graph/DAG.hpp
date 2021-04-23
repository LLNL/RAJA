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

#ifndef RAJA_policy_loop_graph_DAG_HPP
#define RAJA_policy_loop_graph_DAG_HPP

#include "RAJA/config.hpp"

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
struct DAGExec<loop_graph, GraphResource>
{
  resources::EventProxy<GraphResource> operator()(
      DAG<loop_graph, GraphResource>& dag, GraphResource& gr)
  {
    // exec all nodes in a correct order
    dag.forward_traverse(
          [](Node<GraphResource>*) {
            // do nothing
          },
          [&](Node<GraphResource>* node) {
            node->exec(gr);
          },
          [](Node<GraphResource>*) {
            // do nothing
          });
    return resources::EventProxy<GraphResource>(&gr);
  }
};

}  // namespace detail

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif
