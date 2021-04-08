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

#ifndef RAJA_pattern_graph_DAG_HPP
#define RAJA_pattern_graph_DAG_HPP

#include "RAJA/config.hpp"
#include "RAJA/pattern/graph/Node.hpp"
#include "RAJA/util/macros.hpp"

namespace RAJA
{

namespace expt
{

namespace graph
{

template < typename policy >
struct DAG
{
  using Resource = typename resources::get_resource<policy>::type;

  RAJA_INLINE
  DAG() = default;

  bool empty() const
  {
    return (m_root == nullptr);
  }

  void insert_node(Node* node)
  {
    if (m_root == nullptr) {
      m_root = node;
    } else {
      RAJA_ABORT_OR_THROW("DAG::insert_node");
    }
  }

  void exec(Resource& r);

  ~DAG()
  {
    if (m_root != nullptr) {
      delete m_root; m_root = nullptr;
    }
  }

private:
  Node* m_root = nullptr;
};

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif
