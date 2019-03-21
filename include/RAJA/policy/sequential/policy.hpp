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
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
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



}  // namespace RAJA

#endif
