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
 * \brief   Header file containing RAJA headers for sequential execution.
 *
 *          These methods work on all platforms.
 *
 ******************************************************************************
 */

#ifndef RAJA_sequential_HXX
#define RAJA_sequential_HXX

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
struct seq_exec {};

///
/// Index set segment iteration policies
///
struct seq_segit {};

///
///////////////////////////////////////////////////////////////////////
///
/// Reduction execution policies
///
///////////////////////////////////////////////////////////////////////
///
struct seq_reduce {};

}  // closing brace for RAJA namespace


#include "reduce_sequential.hxx"
#include "forall_sequential.hxx"

#endif  // closing endif for header file include guard

