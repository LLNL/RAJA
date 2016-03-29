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
 * \brief   Implementation file for routines used to manage
 *          memory for CUDA reductions and other operations.
 *
 ******************************************************************************
 */

#include "RAJA/exec-cuda/MemUtils_CUDA.hxx"

#include "RAJA/int_datatypes.hxx"

#include "RAJA/reducers.hxx"

#if defined(RAJA_USE_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>

#include<string>
#include<iostream>

namespace RAJA {

//
// Static array used to keep track of which unique ids 
// for CUDA reduction objects are used and which are not.
//
static bool cuda_reduction_id_used[RAJA_MAX_REDUCE_VARS];

//
// Pointer to hold shared managed memory block for RAJA-Cuda reductions.
//
CudaReductionBlockDataType* s_cuda_reduction_mem_block = 0;

/*
*************************************************************************
*
* Return next available valid reduction id, or complain and exit if
* no valid id is available.
*
*************************************************************************
*/
int getCudaReductionId()
{
   static int first_time_called = true;

   if (first_time_called) {

      for (int id = 0; id < RAJA_MAX_REDUCE_VARS; ++id) {
         cuda_reduction_id_used[id] = false;
      }

      first_time_called = false;
   }

   int id = 0;
   while ( id < RAJA_MAX_REDUCE_VARS && cuda_reduction_id_used[id] ) {
     id++;
   }

   if ( id >= RAJA_MAX_REDUCE_VARS ) {
      std::cerr << "\n Exceeded allowable RAJA CUDA reduction count, "
                << "FILE: "<< __FILE__ << " line: "<< __LINE__ << std::endl;
      exit(1);
   }

   cuda_reduction_id_used[id] = true;

   return id;
}


/*
*************************************************************************
*
* Release given redution id and make inactive.
*
*************************************************************************
*/
void releaseCudaReductionId(int id)
{
   if ( id < RAJA_MAX_REDUCE_VARS ) {
      cuda_reduction_id_used[id] = false;
   }
}

/*
*************************************************************************
*
* Return pointer into shared RAJA-CUDA managed reduction memory block
* for reducer object with given id. Allocate block if not already allocated.
*
*************************************************************************
*/
CudaReductionBlockDataType* getCudaReductionMemBlock(int id)
{
   //
   // For each reducer object, we want a chunk of managed memory that
   // holds RAJA_CUDA_REDUCE_BLOCK_LENGTH slots for the reduction 
   // value for each thread, a single slot for the global reduced value
   // across grid blocks, and a single slot for the max grid size.  
   //
   int block_offset = RAJA_CUDA_REDUCE_BLOCK_LENGTH + 1 + 1 + 1;

   if (s_cuda_reduction_mem_block == 0) {
      int len = RAJA_MAX_REDUCE_VARS * block_offset;

      cudaError_t cudaerr = 
         cudaMallocManaged((void **)&s_cuda_reduction_mem_block,
                           sizeof(CudaReductionBlockDataType)*len,
                           cudaMemAttachGlobal);

      if ( cudaerr != cudaSuccess ) {
         std::cerr << "\n ERROR in cudaMallocManaged call, FILE: "
                   << __FILE__ << " line " << __LINE__ << std::endl;
         exit(1);
      }
      cudaMemset(s_cuda_reduction_mem_block, 0, 
                 sizeof(CudaReductionBlockDataType)*len);

      atexit(freeCudaReductionMemBlock);
   }

   return &(s_cuda_reduction_mem_block[id * block_offset]) ;
}

/*
*************************************************************************
*
* Free managed memory blocks used in RAJA-Cuda reductions.
*
*************************************************************************
*/
void freeCudaReductionMemBlock()
{
   if ( s_cuda_reduction_mem_block != 0 ) {
      cudaError_t cudaerr = cudaFree(s_cuda_reduction_mem_block);
      s_cuda_reduction_mem_block = 0;
      if (cudaerr != cudaSuccess) {
         std::cerr << "\n ERROR in cudaFree call, FILE: "
                   << __FILE__ << " line " << __LINE__ << std::endl;
         exit(1);
       }
   }
}


}  // closing brace for RAJA namespace

#endif  // if defined(RAJA_USE_CUDA)
