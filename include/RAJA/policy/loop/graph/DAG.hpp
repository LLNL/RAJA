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

template <>
void DAG<loop_graph>::exec(typename DAG<loop_graph>::Resource&)
{
  // exec all nodes in a correct order
  forward_traverse(
        [](Node* node) {
          node->exec();
        },
        [](Node*) {
          // do nothing
        });
}

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif
