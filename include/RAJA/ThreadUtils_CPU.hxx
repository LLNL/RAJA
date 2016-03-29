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
 * \brief   Header file defining prototypes for routines used to manage
 *          CPU threading operations.
 *
 ******************************************************************************
 */

#ifndef RAJA_ThreadUtils_CPU_HXX
#define RAJA_ThreadUtils_CPU_HXX

#include "config.hxx"

namespace RAJA {

/*!
*************************************************************************
*
* Return max number of available threads for code run on CPU.
*
*************************************************************************
*/
int getMaxThreadsCPU(); 


}  // closing brace for RAJA namespace


#endif  // closing endif for header file include guard
