/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA headers for tbb execution.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_tbb_HPP
#define RAJA_tbb_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_TBB)

#include "RAJA/policy/tbb/forall.hpp"
#include "RAJA/policy/tbb/policy.hpp"
#include "RAJA/policy/tbb/reduce.hpp"
#include "RAJA/policy/tbb/scan.hpp"
#include "RAJA/policy/tbb/sort.hpp"
#include "RAJA/policy/tbb/WorkGroup.hpp"

#endif

#endif  // closing endif for header file include guard
