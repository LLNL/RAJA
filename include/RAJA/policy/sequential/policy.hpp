/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA sequential policy definitions.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef policy_sequential_HPP
#define policy_sequential_HPP

#include "RAJA/policy/PolicyBase.hpp"

namespace RAJA
{
namespace policy
{
namespace sequential
{

//
//////////////////////////////////////////////////////////////////////
//
// Execution policies
//
//////////////////////////////////////////////////////////////////////
//

///
/// Segment execution policies
///

struct seq_region : make_policy_pattern_launch_platform_t<Policy::sequential,
                                                          Pattern::region,
                                                          Launch::sync,
                                                          Platform::host> {
};

struct seq_launch_t : make_policy_pattern_launch_platform_t<Policy::sequential,
                                                            Pattern::region,
                                                            Launch::sync,
                                                            Platform::host> {
};

struct seq_exec : make_policy_pattern_launch_platform_t<Policy::sequential,
                                                        Pattern::forall,
                                                        Launch::undefined,
                                                        Platform::host> {
};

///
/// Index set segment iteration policies
///
using seq_segit = seq_exec;

///
/// WorkGroup execution policies
///
struct seq_work : make_policy_pattern_launch_platform_t<Policy::sequential,
                                                        Pattern::workgroup_exec,
                                                        Launch::sync,
                                                        Platform::host> {
};

///
/// graph execution policies
///
struct seq_graph : make_policy_pattern_launch_platform_t<Policy::sequential,
                                                         Pattern::graph,
                                                         Launch::sync,
                                                         Platform::host> {
};

///
///////////////////////////////////////////////////////////////////////
///
/// Reduction execution policies
///
///////////////////////////////////////////////////////////////////////
///
struct seq_reduce : make_policy_pattern_launch_platform_t<Policy::sequential,
                                                          Pattern::forall,
                                                          Launch::undefined,
                                                          Platform::host> {
};
}  // namespace sequential
}  // namespace policy

using policy::sequential::seq_exec;
using policy::sequential::seq_reduce;
using policy::sequential::seq_region;
using policy::sequential::seq_segit;
using policy::sequential::seq_work;
using policy::sequential::seq_graph;

namespace expt
{
  using policy::sequential::seq_launch_t;
}


}  // namespace RAJA

#endif
