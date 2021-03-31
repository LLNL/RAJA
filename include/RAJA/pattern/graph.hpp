/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing headers for RAJA::graph backends
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_graph_HPP
#define RAJA_pattern_graph_HPP

#include "RAJA/pattern/graph/DAG.hpp"

//
// All platforms must support host execution.
//
#include "RAJA/policy/sequential/policy.hpp"
#include "RAJA/policy/loop/policy.hpp"

// #if defined(RAJA_ENABLE_OPENMP)
// #include "RAJA/policy/openmp/graph.hpp"
// #endif

#endif /* RAJA_pattern_graph_HPP */
