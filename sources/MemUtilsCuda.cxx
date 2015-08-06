/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for operations used to manage
 *          memory for CUDA execution methods.
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#include "RAJA/MemUtilsCuda.hxx"

#include <iostream>
#include <cstdlib>

#if defined(RAJA_USE_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>

namespace RAJA {

// 
// Pointer to hold managed memory block for RAJA-CUDA reductions.
//
void* s_cuda_reduction_mem_block = 0;

/*
*************************************************************************
*
* Allocate managed memory block for RAJA-CUDA reductions if it doesn't
* exist; else a no-op. In either case, pointer to block is returned. 
*
*************************************************************************
*/
void* allocallocCudaReductionMemBlockData(int BLOCK_LENGTH, int MAX_REDUCE_VARS)
{
   if (s_cuda_reduction_mem_block == 0) {
      cudaError_t cudaerr = 
         cudaMallocManaged((void **)&s_cuda_reduction_mem_block,
                           sizeof(CudaReduceBlockAllocType) * 
                           (BLOCK_LENGTH * MAX_REDUCE_VARS),
                           cudaMemAttachGlobal);

      if ( cudaerr != cudaSuccess ) {
         std::cerr << "\n ERROR in cudaMallocManaged call, FILE: "
                   << __FILE__ << " line " << __LINE__ << std::endl;
         exit(1);
      }
   }

   atexit(freeCudaReductionMemBlockData);

   return s_cuda_reduction_mem_block;
}


/*
*************************************************************************
*
* Free managed memory block used in RAJA-CUDA reductions.
*
*************************************************************************
*/
void freeCudaReductionMemBlockData()
{
   if ( s_cuda_reduction_mem_block != 0 ) {
      cudaError_t cudaerr = cudaFree(s_cuda_reduction_mem_block);
      if (cudaerr != cudaSuccess) {
         std::cerr << "\n ERROR in cudaFree call, FILE: "
                      << __FILE__ << " line " << __LINE__ << std::endl;
         exit(1);
       }
   }
}

/*
 ******************************************************************************
 *
 * Return pointer to managed memory block for RAJA-CUDA reductions.
 *
 ******************************************************************************
 */
void* getCudaReductionMemBlockDataVoidPtr()
{
   return s_cuda_reduction_mem_block;
}


}  // closing brace for RAJA namespace

#endif  // #if defined(RAJA_USE_CUDA)
