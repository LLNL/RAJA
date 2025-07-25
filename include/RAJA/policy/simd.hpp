/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA headers for SIMD segment execution.
 *
 *          These methods work on all platforms.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_simd_HPP
#define RAJA_simd_HPP

#include "RAJA/policy/simd/forall.hpp"
#include "RAJA/policy/simd/policy.hpp"
#include "RAJA/policy/sequential/launch.hpp"
#include "RAJA/policy/simd/kernel/For.hpp"
#include "RAJA/policy/simd/kernel/ForICount.hpp"
#include "RAJA/policy/simd/params/reduce.hpp"
#include "RAJA/policy/simd/params/kernel_name.hpp"

#endif  // closing endif for header file include guard
