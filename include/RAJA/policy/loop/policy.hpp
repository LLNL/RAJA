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
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
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

#ifndef policy_loop_HPP
#define policy_loop_HPP

#include "RAJA/policy/PolicyBase.hpp"

#include "RAJA/policy/sequential/policy.hpp"

namespace RAJA
{
namespace policy
{
namespace loop
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

struct loop_exec : make_policy_pattern_launch_platform_t<Policy::loop,
                                                         Pattern::forall,
                                                         Launch::undefined,
                                                         Platform::host> {
};

///
/// Index set segment iteration policies
///
using loop_segit = loop_exec;

///
///////////////////////////////////////////////////////////////////////
///
/// Reduction execution policies
///
///////////////////////////////////////////////////////////////////////
///
using loop_reduce = seq_reduce;

}  // end namespace loop

}  // end namespace policy

using policy::loop::loop_exec;
using policy::loop::loop_segit;
using policy::loop::loop_reduce;

}  // closing brace for RAJA namespace

#endif
