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
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_teams_HPP
#define RAJA_pattern_teams_HPP

#include "RAJA/pattern/teams/teams_core.hpp"

//
// All platforms must support loop execution.
//
#include "RAJA/policy/loop/teams.hpp"

#if defined(RAJA_CUDA_ACTIVE)
#include "RAJA/policy/cuda/teams.hpp"
#endif

#if defined(RAJA_ENABLE_HIP)
#include "RAJA/policy/hip/teams.hpp"
#endif

#if defined(RAJA_ENABLE_OPENMP)
#include "RAJA/policy/openmp/teams.hpp"
#endif

#endif /* RAJA_pattern_teams_HPP */
