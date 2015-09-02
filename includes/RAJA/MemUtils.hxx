/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file defining prototypes for operations used to manage
 *          memory for reductions and other operations.
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#ifndef RAJA_MemUtils_HXX
#define RAJA_MemUtils_HXX

#include "reducers.hxx"

#include <cstdio>

namespace RAJA {

//////////////////////////////////////////////////////////////////////
//
// Utilities and methods for CPU reductions.
//
//////////////////////////////////////////////////////////////////////

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


#if defined(RAJA_USE_CUDA)

//////////////////////////////////////////////////////////////////////
//
// Utilities and methods for CUDA reductions.
//
//////////////////////////////////////////////////////////////////////


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
*************************************************************************
*
* Set current CUDA grid size used in forall methods as given arg value
* so it can be used in other methods (i.e., reduction finalization).
*
*************************************************************************
*/
void setCurrentGridSize(size_t s);

/*!
*************************************************************************
*
* Retrieve current CUDA grid size value.
*
*************************************************************************
*/
size_t getCurrentGridSize();

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
 *            (RAJA_CUDA_REDUCE_BLOCK_LENGTH * RAJA_MAX_REDUCE_VARS +
 *                                             RAJA_MAX_REDUCE_VARS)
 *
 ******************************************************************************
 */
CudaReductionBlockDataType* getCudaReductionMemBlock();

/*!
 ******************************************************************************
 *
 * \brief  Free managed memory block used in RAJA-Cuda reductions.
 *
 ******************************************************************************
 */
void freeCudaReductionMemBlock();

/*!
*************************************************************************
*
* Return offset into shared RAJA-Cuda reduction memory block for
* reduction object with given id.
*
*************************************************************************
*/
int getCudaReductionMemBlockOffset(int id);


#endif

}  // closing brace for RAJA namespace


#endif  // closing endif for header file include guard
