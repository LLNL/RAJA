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
 * \brief   Header file defining prototypes for routines used to manage
 *          memory for CUDA reductions and other operations.
 *
 ******************************************************************************
 */

#ifndef RAJA_MemUtils_CUDA_HXX
#define RAJA_MemUtils_CUDA_HXX

#include "RAJA/config.hxx"


#if defined(RAJA_USE_CUDA)

namespace RAJA {


#define RAJA_CUDA_REDUCE_BLOCK_LENGTH (1024 + 8) * 16

///
/// Typedef defining common data type for RAJA-Cuda reduction data blocks
/// (use this in all cases to avoid type confusion).
///
typedef double CudaReductionBlockDataType;

/*!
*************************************************************************
*
* Return next available valid reduction id, or complain and exit if
* no valid id is available.
*
*************************************************************************
*/
int getCudaReductionId();

/*!
*************************************************************************
*
* Release given redution id and make inactive.
*
*************************************************************************
*/
void releaseCudaReductionId(int id);

/*!
 ******************************************************************************
 *
 * \brief  Return pointer into shared memory block for RAJA-Cuda reduction
 *         with given id.
 *
 *         Allocates data block if it isn't allocated already.
 *
 * NOTE: Block size will be:
 *
 *          sizeof(CudaReductionBlockDataType) * 
 *            RAJA_MAX_REDUCE_VARS * ( RAJA_CUDA_REDUCE_BLOCK_LENGTH + 1 + 1 )
 *
 *       For each reducer object, we want a chunk of managed memory that
 *       holds RAJA_CUDA_REDUCE_BLOCK_LENGTH slots for the reduction
 *       value for each thread, a single slot for the global reduced value
 *       across grid blocks, and a single slot for the max grid size
 *
 ******************************************************************************
 */
CudaReductionBlockDataType* getCudaReductionMemBlock(int id);

/*!
 ******************************************************************************
 *
 * \brief  Free managed memory block used in RAJA-Cuda reductions.
 *
 ******************************************************************************
 */
void freeCudaReductionMemBlock();


}  // closing brace for RAJA namespace


#endif  // closing endif for RAJA_USE_CUDA

#endif  // closing endif for header file include guard
