/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * For release details and restrictions, please see raja/README-license.txt
 */

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

#include "RAJA/ThreadUtils_CPU.hxx"

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(RAJA_USE_CILK)
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#endif

#include <algorithm>

namespace RAJA {

/*
*************************************************************************
*
* Return max number of available threads for code run on CPU.
*
*************************************************************************
*/
int getMaxThreadsCPU()
{
   int nthreads = 1;

#if defined(_OPENMP)
   nthreads = omp_get_max_threads();
#endif
#if defined(RAJA_USE_CILK)
   int nworkers = __cilkrts_get_nworkers();
   nthreads = std::max(nthreads, nworkers);
#endif

   return nthreads;
}


}  // closing brace for RAJA namespace
