/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA headers for OpenMP execution.
 *
 *          These methods work only on platforms that support OpenMP.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
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

#ifndef RAJA_openmp_target_HPP
#define RAJA_openmp_target_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_OPENMP) && defined(RAJA_ENABLE_TARGET_OPENMP)

#include <omp.h>

#include "RAJA/policy/openmp_target/policy.hpp"
#include "RAJA/policy/openmp_target/kernel.hpp"
#include "RAJA/policy/openmp_target/forall.hpp"
#include "RAJA/policy/openmp_target/reduce.hpp"

#endif  // closing endif for if defined(RAJA_ENABLE_OPENMP) && defined(RAJA_ENABLE_TARGET_OPENMP)

#endif  // closing endif for header file include guard
