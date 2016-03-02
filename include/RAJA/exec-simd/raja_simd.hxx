/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and
 * further review from Lawrence Livermore National Laboratory.
 */

/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA headers for SIMD segment execution.
 *
 *          These methods work on all platforms.
 *
 ******************************************************************************
 */

#ifndef RAJA_simd_HXX
#define RAJA_simd_HXX

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
struct simd_exec {};

//
// NOTE: There is no Index set segment iteration policy for SIMD
//


///
///////////////////////////////////////////////////////////////////////
///
/// Reduction execution policies
///
///////////////////////////////////////////////////////////////////////
///

//
// NOTE: RAJA reductions in SIMD loops use seg_reduce policy
//

#include "forall_simd.hxx"

#endif  // closing endif for header file include guard

