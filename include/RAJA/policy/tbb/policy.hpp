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

}  // closing brace for tbb
}  // closing brace for policy

using policy::tbb::tbb_for_exec;
using policy::tbb::tbb_for_static;
using policy::tbb::tbb_for_dynamic;
using policy::tbb::tbb_segit;
using policy::tbb::tbb_reduce;

}  // closing brace for RAJA namespace

#endif
