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
 *          memory for CPU reductions and other operations.
 *
 ******************************************************************************
 */

#ifndef RAJA_MemUtils_CPU_HXX
#define RAJA_MemUtils_CPU_HXX

#include "config.hxx"

#include "int_datatypes.hxx"


namespace RAJA {

///
/// Typedef defining common data type for RAJA-CPU reduction data blocks
/// (use this in all cases to avoid type confusion).
///
typedef double CPUReductionBlockDataType;

/*!
*************************************************************************
*
* Return available valid reduction id and record reduction type for
* that id, or complain and exit if no ids are available.
*
*************************************************************************
*/
int getCPUReductionId();

/*!
*************************************************************************
*
* Release given redution id so it can be reused.
*
*************************************************************************
*/
void releaseCPUReductionId(int id);

/*!
 ******************************************************************************
 *
 * \brief  Return pointer into shared memory block for RAJA-CPU reduction
 *         data for reduction object with given id.
 *
 *         Allocates data block if it isn't allocated already.
 *
 * NOTE: Block size will be of one of the following sizes:
 *
 *       When compiled with OpenMP : 
 * 
 *          omp_get_max_threads() * MAX_REDUCE_VARS_CPU *
 *          COHERENCE_BLOCK_SIZE/sizeof(CPUReductionBlockDataType)
 *
 *       When compiled without OpenMP :
 *
 *          MAX_REDUCE_VARS_CPU *
 *          COHERENCE_BLOCK_SIZE/sizeof(CPUReductionBlockDataType)
 *
 ******************************************************************************
 */
CPUReductionBlockDataType* getCPUReductionMemBlock(int id);

/*!
 ******************************************************************************
 *
 * \brief  Free managed memory block used in RAJA-CPU reductions.
 *
 ******************************************************************************
 */
void freeCPUReductionMemBlock();

/*!
 ******************************************************************************
 *
 * \brief  Return pointer into shared memory block for index location in
 *         RAJA-CPU "loc" reductions for reduction object with given id.
 *
 *         Allocates data block if it isn't allocated already.
 *
 * NOTE: Block size will be of one of the following sizes:
 *
 *       When compiled with OpenMP :
 *
 *          omp_get_max_threads() * MAX_REDUCE_VARS_CPU *
 *          COHERENCE_BLOCK_SIZE/sizeof(Index_type)
 *
 *       When compiled without OpenMP :
 *
 *          MAX_REDUCE_VARS_CPU *  
 *          COHERENCE_BLOCK_SIZE/sizeof(Index_type)
 *
 ******************************************************************************
 */
Index_type* getCPUReductionLocBlock(int id);

/*!
 ******************************************************************************
 *
 * \brief  Free managed memory location index block used in RAJA-CPU reductions.
 *
 ******************************************************************************
 */
void freeCPUReductionLocBlock();


}  // closing brace for RAJA namespace


#endif  // closing endif for header file include guard
