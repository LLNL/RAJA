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

#ifndef RAJA_pattern_graph_Collection_HPP
#define RAJA_pattern_graph_Collection_HPP

#include "RAJA/config.hpp"

namespace RAJA
{

namespace expt
{

namespace graph
{

namespace detail
{

struct CollectionArgs { };

struct Collection
{
  Collection() = delete;

  Collection(Collection const&) = delete;
  Collection(Collection&&) = delete;

  Collection& operator=(Collection const&) = delete;
  Collection& operator=(Collection&&) = delete;

  Collection(size_t id)
    : m_my_id(id)
  {
  }

  virtual ~Collection() = default;

  size_t get_my_id() const
  {
    return m_my_id;
  }

  void set_my_id(size_t id)
  {
    m_my_id = id;
  }

  size_t num_nodes() const
  {
    return m_num_nodes;
  }

protected:
  size_t m_my_id;
  size_t m_num_nodes = 0;
};

}  // namespace detail

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif
