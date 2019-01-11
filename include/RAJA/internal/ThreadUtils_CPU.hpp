/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing prototypes and methods for managing
 *          CPU threading operations.
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

#ifndef RAJA_ThreadUtils_CPU_HPP
#define RAJA_ThreadUtils_CPU_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_OPENMP)
#include <omp.h>
#endif

namespace RAJA
{

/*!
*************************************************************************
*
* Return max number of available OpenMP threads.
*
*************************************************************************
*/
RAJA_INLINE
int getMaxOMPThreadsCPU()
{
  int nthreads = 1;

#if defined(RAJA_ENABLE_OPENMP)
  nthreads = omp_get_max_threads();
#endif

  return nthreads;
}

}  // namespace RAJA

#endif  // closing endif for header file include guard
