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
 * \brief   Header file containing RAJA headers for Intel CilkPlus execution.
 *
 *          These methods work only on platforms that support Cilk Plus. 
 *
 ******************************************************************************
 */

#ifndef RAJA_cilk_HXX
#define RAJA_cilk_HXX

#if defined(RAJA_USE_CILK)

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
struct cilk_for_exec {};

///
/// Index set segment iteration policies
///
struct cilk_for_segit {};

///
///////////////////////////////////////////////////////////////////////
///
/// Reduction execution policies
///
///////////////////////////////////////////////////////////////////////
///
struct cilk_reduce {};

}  // closing brace for RAJA namespace


#include "reduce_cilk.hxx"
#include "forall_cilk.hxx"


#endif  // closing endif for if defined(RAJA_USE_CILK)

#endif  // closing endif for header file include guard

