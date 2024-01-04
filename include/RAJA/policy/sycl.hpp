/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA headers for SYCL execution.
 *
 *          These methods work only on platforms that support SYCL.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_sycl_HPP
#define RAJA_sycl_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_SYCL_ACTIVE)

#include <CL/sycl.hpp>

#include "RAJA/policy/sycl/forall.hpp"
#include "RAJA/policy/sycl/policy.hpp"
#include "RAJA/policy/sycl/reduce.hpp"
//#include "RAJA/policy/sycl/scan.hpp"
//#include "RAJA/policy/sycl/sort.hpp"
#include "RAJA/policy/sycl/kernel.hpp"
//#include "RAJA/policy/sycl/synchronize.hpp"
#include "RAJA/policy/sycl/launch.hpp"
//#include "RAJA/policy/sycl/WorkGroup.hpp"

#endif  // closing endif for if defined(RAJA_ENABLE_SYCL)

#endif  // closing endif for header file include guard
