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
namespace RAJA {

struct simd_exec {};

}

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

