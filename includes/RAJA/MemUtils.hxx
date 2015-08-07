/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file defining prototypes for operations used to manage
 *          memory for CPU reductions and other operations.
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#ifndef RAJA_MemUtils_HXX
#define RAJA_MemUtils_HXX

#include "reducers.hxx"

#if defined(_OPENMP)
#include <omp.h>
#endif


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
int getCPUReductionId(ReductionType type);

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
 *         with given id.
 *
 *         Allocates data block if it isn't allocated already.
 *
 * NOTE: Block size will be of one of the following sizes:
 *
 *       When compiled with OpenMP : 
 * 
 *          sizeof(CPUReductionBlockDataType) * omp_get_max_threads() *
 *          MAX_REDUCE_VARS_CPU
 *
 *       When compiled without OpenMP :
 *
 *          sizeof(CPUReductionBlockDataType) * MAX_REDUCE_VARS_CPU
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
 * \brief  Set value in RAJA-CPU initial value shared memory block for
 *         reduction object with given id. 
 *
 *         Allocates data block if it isn't allocated already.
 *
 * NOTE: Block size will be of size
 *       sizeof(CPUReduceBlockAllocType) * MAX_REDUCE_VARS_CPU
 *
 ******************************************************************************
 */
void setCPUReductionInitValue(int id, CPUReductionBlockDataType val);

/*!
 ******************************************************************************
 *
 * \brief  Get value in RAJA-CPU initial value shared memory block for
 *         reduction object with given id.
 *
 ******************************************************************************
 */
CPUReductionBlockDataType getCPUReductionInitValue(int id);

/*!
 ******************************************************************************
 *
 * \brief  Free shared memory block for initial values used in RAJA-CPU 
 *         reductions.
 *
 ******************************************************************************
 */
void freeCPUReductionInitData();

}  // closing brace for RAJA namespace


#endif  // closing endif for header file include guard
