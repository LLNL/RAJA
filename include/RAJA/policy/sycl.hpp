/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA headers for sycl execution.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_SYCL_HPP
#define RAJA_SYCL_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "RAJA/policy/sycl/forall.hpp"
#include "RAJA/policy/sycl/policy.hpp"
//#include "RAJA/policy/sycl/reduce.hpp"
//#include "RAJA/policy/sycl/scan.hpp"
#include "RAJA/policy/sycl/kernel.hpp"

#endif

#endif  // closing endif for header file include guard
