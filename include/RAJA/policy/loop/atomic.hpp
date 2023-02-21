/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining loop atomic operations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_loop_atomic_HPP
#define RAJA_policy_loop_atomic_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#include "RAJA/policy/loop/policy.hpp"
#include "RAJA/policy/sequential/atomic.hpp"

namespace RAJA
{

// backwards compatability
using loop_atomic = policy::loop::loop_atomic;

}  // namespace RAJA

#endif  // guard
