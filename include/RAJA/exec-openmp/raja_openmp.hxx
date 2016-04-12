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
 * \brief   Header file containing RAJA headers for OpenMP execution.
 *
 *          These methods work only on platforms that support OpenMP. 
 *
 ******************************************************************************
 */

#ifndef RAJA_openmp_HXX
#define RAJA_openmp_HXX


#if defined(RAJA_USE_OPENMP)

namespace RAJA {

//
//////////////////////////////////////////////////////////////////////
//
// Execution policies
//
//////////////////////////////////////////////////////////////////////
//

///
/// Segment execution policies
///
struct omp_parallel_for_exec {};
//struct omp_parallel_for_nowait_exec {};
struct omp_for_nowait_exec {};

///
/// Index set segment iteration policies
///
struct omp_parallel_for_segit {};
struct omp_parallel_segit {};
struct omp_taskgraph_segit {};
struct omp_taskgraph_interval_segit {};

///
///////////////////////////////////////////////////////////////////////
///
/// Reduction execution policies
///
///////////////////////////////////////////////////////////////////////
///
struct omp_reduce {};

}  // closing brace for RAJA namespace


#include "reduce_openmp.hxx"
#include "forall_openmp.hxx"

#if defined(RAJA_USE_TILE)
#include "forallN_openmp.hxx"
#endif

#endif  // closing endif for if defined(RAJA_USE_OPENMP)

#endif  // closing endif for header file include guard

