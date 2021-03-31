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

#ifndef RAJA_pattern_graph_Node_HPP
#define RAJA_pattern_graph_Node_HPP

#include "RAJA/config.hpp"

namespace RAJA
{

namespace expt
{

namespace graph
{

struct Node
{
  RAJA_INLINE
  Node() = default;

  virtual void exec() = 0;

  virtual ~Node() = default;
};

}  // namespace graph

}  // namespace expt

}  // namespace RAJA
#endif
