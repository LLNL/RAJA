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
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
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
#include "RAJA/pattern/teams/teams_sequential.hpp"

#if defined(RAJA_CUDA_ACTIVE)
#include "RAJA/pattern/teams/teams_cuda.hpp"
#endif

#if defined(RAJA_ENABLE_HIP)
#include "RAJA/pattern/teams/teams_hip.hpp"
#endif

#if defined(RAJA_ENABLE_OPENMP)
#include "RAJA/pattern/teams/teams_openmp.hpp"
#endif

#endif /* RAJA_pattern_teams_HPP */
