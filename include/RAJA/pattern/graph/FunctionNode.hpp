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

#ifndef RAJA_pattern_graph_FunctionNode_HPP
#define RAJA_pattern_graph_FunctionNode_HPP

#include "RAJA/config.hpp"

#include <utility>
#include <type_traits>

#include "RAJA/pattern/graph/DAG.hpp"
#include "RAJA/pattern/graph/Node.hpp"

namespace RAJA
{

namespace expt
{

namespace graph
{

template < typename FunctionBody >
struct FunctionNode;

namespace detail
{

template < typename FunctionBody >
struct FunctionArgs : NodeArgs
{
  using node_type = FunctionNode<FunctionBody>;

  template < typename Func >
  FunctionArgs(Func&& func)
    : m_function(std::forward<Func>(func))
  {
  }

  FunctionBody m_function;
};

}  // namespace detail

template < typename Func >
RAJA_INLINE detail::FunctionArgs<camp::decay<Func>>
Function(Func&& func)
{
  return detail::FunctionArgs<camp::decay<Func>>(std::forward<Func>(func));
}

template < typename FunctionBody >
struct FunctionNode : detail::NodeData
{
  using function_type = FunctionBody;
  using resource = resources::Host;
  using args_type = detail::FunctionArgs<FunctionBody>;

  FunctionNode() = delete;

  FunctionNode(FunctionNode const&) = delete;
  FunctionNode(FunctionNode&&) = delete;

  FunctionNode& operator=(FunctionNode const&) = delete;
  FunctionNode& operator=(FunctionNode&&) = delete;

  FunctionNode(args_type const& args)
    : m_function(args.m_function)
  {
  }
  FunctionNode(args_type&& args)
    : m_function(std::move(args.m_function))
  {
  }

  FunctionNode& operator=(args_type const& args)
  {
    m_function = args.m_function;
    return *this;
  }
  FunctionNode& operator=(args_type&& args)
  {
    m_function = std::move(args.m_function);
    return *this;
  }

  virtual ~FunctionNode() = default;

  FunctionBody const& get_function() const
  {
    return m_function;
  }

  template < typename Func >
  void set_function(Func&& func)
  {
    m_function = std::forward<Func>(func);
  }

protected:
  void exec() override
  {
    m_function();
  }

private:
  FunctionBody m_function;
};

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif
