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
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef policy_tbb_HPP
#define policy_tbb_HPP

#include "RAJA/policy/PolicyBase.hpp"

#include <cstddef>

namespace RAJA
{
namespace policy
{
namespace tbb
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

struct tbb_for_dynamic
    : make_policy_pattern_launch_platform_t<Policy::tbb,
                                            Pattern::forall,
                                            Launch::undefined,
                                            Platform::host> {
  std::size_t grain_size;
  tbb_for_dynamic(std::size_t grain_size_ = 1) : grain_size(grain_size_) {}
};


template <std::size_t GrainSize = 1>
struct tbb_for_static : make_policy_pattern_launch_platform_t<Policy::tbb,
                                                              Pattern::forall,
                                                              Launch::undefined,
                                                              Platform::host> {
};

using tbb_for_exec = tbb_for_static<>;

///
/// Index set segment iteration policies
///
using tbb_segit = tbb_for_exec;

///
/// WorkGroup execution policies
///
struct tbb_work : make_policy_pattern_launch_platform_t<Policy::tbb,
                                                        Pattern::workgroup_exec,
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
struct tbb_reduce : make_policy_pattern_launch_platform_t<Policy::tbb,
                                                          Pattern::reduce,
                                                          Launch::undefined,
                                                          Platform::host> {
};

}  // namespace tbb
}  // namespace policy

using policy::tbb::tbb_for_dynamic;
using policy::tbb::tbb_for_exec;
using policy::tbb::tbb_for_static;
using policy::tbb::tbb_reduce;
using policy::tbb::tbb_segit;
using policy::tbb::tbb_work;

}  // namespace RAJA

#endif
