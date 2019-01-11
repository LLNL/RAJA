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


#ifndef RAJA_openmp_HPP
#define RAJA_openmp_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_OPENMP)

#include <omp.h>
#include <iostream>
#include <thread>

#include "RAJA/policy/openmp/atomic.hpp"
#include "RAJA/policy/openmp/forall.hpp"
#include "RAJA/policy/openmp/kernel.hpp"
#include "RAJA/policy/openmp/policy.hpp"
#include "RAJA/policy/openmp/reduce.hpp"
#include "RAJA/policy/openmp/region.hpp"
#include "RAJA/policy/openmp/scan.hpp"
#include "RAJA/policy/openmp/synchronize.hpp"

#endif  // closing endif for if defined(RAJA_ENABLE_OPENMP)

#endif  // closing endif for header file include guard
