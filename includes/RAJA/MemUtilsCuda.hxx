/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file defining prototypes for operations used to manage
 *          memory for CUDA reduction methods.
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#ifndef RAJA_MemUtilsCuda_HXX
#define RAJA_MemUtilsCuda_HXX

namespace RAJA {

typedef double CudaReduceBlockAllocType;

/*!
 ******************************************************************************
 *
 * \brief  Return pointer to managed memory block for RAJA-CUDA reductions.
 *         Allocates data block if it isn't allocated already.
 *
 * NOTE: Block size will be of size 
 *       sizeof(CudaReduceBlockAllocType) * BLOCK_LENGTH * MAX_REDUCE_VARS
 *
 ******************************************************************************
 */
void* allocCudaReductionMemBlockData(int BLOCK_LENGTH,
                                     int MAX_REDUCE_VARS);

/*!
 ******************************************************************************
 *
 * \brief  Free managed memory block used in RAJA-CUDA reductions.
 *
 ******************************************************************************
 */
void freeCudaReductionMemBlockData();

/*!
 ******************************************************************************
 *
 * \brief  Return pointer to managed memory block for RAJA-CUDA reductions.
 *
 ******************************************************************************
 */
void* getCudaReductionMemBlockDataVoidPtr();

/*!
 ******************************************************************************
 *
 * \brief  Template that returns pointer of particular type to managed memory 
 *         block used in RAJA-CUDA reductions.
 *
 ******************************************************************************
 */
template <typename T> T* getCudaReductionMemBlockData()
{
  return( static_cast<T*>(getCudaReductionMemBlockDataVoidPtr() ) );
}



}  // closing brace for RAJA namespace


#endif  // closing endif for header file include guard
