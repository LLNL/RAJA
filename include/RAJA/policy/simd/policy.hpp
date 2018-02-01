/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA simd policy definitions.
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

#ifndef policy_simd_HPP
#define policy_simd_HPP

#include "RAJA/policy/PolicyBase.hpp"

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
namespace RAJA
{
namespace policy
{
namespace simd
{

struct simd_exec : make_policy_pattern_launch_platform_t<Policy::sequential,
                                                         Pattern::forall,
                                                         Launch::undefined,
                                                         Platform::host> {
};

}  // end of namespace simd

}  // end of namespace policy

using policy::simd::simd_exec;

}  // end of namespace RAJA

#endif
