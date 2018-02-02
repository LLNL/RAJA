/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for routines used to manage
 *          CPU threading operations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
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

#include "RAJA/internal/ThreadUtils_CPU.hpp"

#if defined(RAJA_ENABLE_OPENMP)
#include <omp.h>
#endif

namespace RAJA
{

/*
*************************************************************************
*
* Return max number of available threads for code run on CPU.
*
*************************************************************************
*/
int getMaxReduceThreadsCPU()
{
  int nthreads = 1;

#if defined(RAJA_ENABLE_OPENMP)
  nthreads = omp_get_max_threads();
#endif

  return nthreads;
}

/*
*************************************************************************
*
* Return max number of OpenMP threads for code run on CPU.
*
*************************************************************************
*/
int getMaxOMPThreadsCPU()
{
  int nthreads = 1;

#if defined(RAJA_ENABLE_OPENMP)
  nthreads = omp_get_max_threads();
#endif

  return nthreads;
}

}  // closing brace for RAJA namespace
