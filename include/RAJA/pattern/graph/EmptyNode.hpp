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

#ifndef RAJA_pattern_graph_EmptyNode_HPP
#define RAJA_pattern_graph_EmptyNode_HPP

#include "RAJA/config.hpp"

#include <utility>
#include <type_traits>

#include "RAJA/pattern/graph/Node.hpp"

namespace RAJA
{

namespace expt
{

namespace graph
{

struct EmptyNode;

namespace detail
{

struct EmptyArgs : NodeArgs
{
  using node_type = EmptyNode;
};

}  // namespace detail

RAJA_INLINE detail::EmptyArgs Empty()
{
  return detail::EmptyArgs();
}

struct EmptyNode : detail::NodeData
{
  using args_type = detail::EmptyArgs;

  EmptyNode() = delete;

  EmptyNode(EmptyNode const&) = delete;
  EmptyNode(EmptyNode&&) = delete;

  EmptyNode& operator=(EmptyNode const&) = delete;
  EmptyNode& operator=(EmptyNode&&) = delete;

  EmptyNode(args_type const&)
  {
  }
  EmptyNode(args_type&&)
  {
  }

  EmptyNode& operator=(args_type const&)
  {
    return *this;
  }
  EmptyNode& operator=(args_type&&)
  {
    return *this;
  }

  virtual ~EmptyNode() = default;

  size_t get_num_iterations() const override
  {
    return 0;
  }

protected:
  void exec() override
  {
  }
};

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif
