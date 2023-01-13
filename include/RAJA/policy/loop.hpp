/*!
******************************************************************************
*
* \file
*
* \brief   Header file containing RAJA headers for sequential execution.
*
*          These methods work on all platforms.
*
******************************************************************************
*/

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_loop_HPP
#define RAJA_loop_HPP

#if !defined(RAJA_ENABLE_DESUL_ATOMICS)
    #include "RAJA/policy/loop/atomic.hpp"
#endif

#include "RAJA/policy/loop/forall.hpp"
#include "RAJA/policy/loop/kernel.hpp"
#include "RAJA/policy/loop/policy.hpp"
#include "RAJA/policy/loop/scan.hpp"
#include "RAJA/policy/loop/sort.hpp"
#include "RAJA/policy/loop/launch.hpp"
#include "RAJA/policy/loop/WorkGroup.hpp"

#endif  // closing endif for header file include guard
