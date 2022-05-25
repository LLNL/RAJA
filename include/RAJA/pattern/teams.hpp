/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing headers for RAJA::Teams backends
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_teams_HPP
#define RAJA_pattern_teams_HPP

#include "RAJA/pattern/teams/teams_core.hpp"

//
// All platforms must support host execution.
//
#include "RAJA/policy/sequential/teams.hpp"
#include "RAJA/policy/loop/teams.hpp"
#include "RAJA/policy/simd/teams.hpp"

#if defined(RAJA_CUDA_ACTIVE)
#include "RAJA/policy/cuda/teams.hpp"
#endif

#if defined(RAJA_HIP_ACTIVE)
#include "RAJA/policy/hip/teams.hpp"
#endif

#if defined(RAJA_ENABLE_OPENMP)
#include "RAJA/policy/openmp/teams.hpp"
#endif

//#if defined(RAJA_ENABLE_SYCL)
#include "RAJA/policy/sycl/teams.hpp"
//#endif

#endif /* RAJA_pattern_teams_HPP */
