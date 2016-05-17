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

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read raja/README-license.txt.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

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

CudaReductionLocBlockDataType* s_cuda_reduction_loc_mem_block = 0;

CudaReductionBlockTallyType* s_cuda_reduction_tally_block = 0;
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


/*
*************************************************************************
*
* Return pointer into shared RAJA-CUDA managed reduction memory block
* for reducer object with given id. Allocate block if not already allocated.
*
*************************************************************************
*/
CudaReductionLocBlockDataType* getCudaReductionLocMemBlock(int id)
{
   //
   // For each reducer object, we want a chunk of managed memory that
   // holds RAJA_CUDA_REDUCE_BLOCK_LENGTH slots for the reduction 
   // value for each thread, a single slot for the global reduced value
   // across grid blocks, and a single slot for the max grid size.  
   //
   int block_offset = RAJA_CUDA_REDUCE_BLOCK_LENGTH + 1 + 1 + 1;

   if (s_cuda_reduction_loc_mem_block == 0) {
      int len = RAJA_MAX_REDUCE_VARS * block_offset;

      cudaError_t cudaerr = 
         cudaMallocManaged((void **)&s_cuda_reduction_loc_mem_block,
                           sizeof(CudaReductionLocBlockDataType)*len,
                           cudaMemAttachGlobal);

      if ( cudaerr != cudaSuccess ) {
         std::cerr << "\n ERROR in cudaMallocManaged call, FILE: "
                   << __FILE__ << " line " << __LINE__ << std::endl;
         exit(1);
      }
      cudaMemset(s_cuda_reduction_loc_mem_block, 0, 
                 sizeof(CudaReductionLocBlockDataType)*len);

      atexit(freeCudaReductionLocMemBlock);
   }

   return &(s_cuda_reduction_loc_mem_block[id * block_offset]) ;
}

/*
*************************************************************************
*
* Free managed memory blocks used in RAJA-Cuda reductions.
*
*************************************************************************
*/
void freeCudaReductionLocMemBlock()
{
   if ( s_cuda_reduction_loc_mem_block != 0 ) {
      cudaError_t cudaerr = cudaFree(s_cuda_reduction_loc_mem_block);
      s_cuda_reduction_loc_mem_block = 0;
      if (cudaerr != cudaSuccess) {
         std::cerr << "\n ERROR in cudaFree call, FILE: "
                   << __FILE__ << " line " << __LINE__ << std::endl;
         exit(1);
       }
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
CudaReductionBlockTallyType* getCudaReductionTallyBlock(int id)
{
   if (s_cuda_reduction_tally_block == 0) {
      int len = RAJA_CUDA_REDUCE_TALLY_LENGTH; 

      cudaError_t cudaerr = 
         cudaMallocManaged((void **)&s_cuda_reduction_tally_block,
                           sizeof(CudaReductionBlockTallyType)*len,
                           cudaMemAttachGlobal);

      if ( cudaerr != cudaSuccess ) {
         std::cerr << "\n ERROR in cudaMallocManaged call, FILE: "
                   << __FILE__ << " line " << __LINE__ << std::endl;
         exit(1);
      }
      cudaMemset(s_cuda_reduction_tally_block, 0, 
                 sizeof(CudaReductionBlockTallyType)*len);

      atexit(freeCudaReductionTallyBlock);
   }

   return &(s_cuda_reduction_tally_block[id]) ;
}

/*
*************************************************************************
*
* Free managed memory blocks used in RAJA-Cuda reductions.
*
*************************************************************************
*/

void freeCudaReductionTallyBlock()
{
   if ( s_cuda_reduction_tally_block != 0 ) {
      cudaError_t cudaerr = cudaFree(s_cuda_reduction_tally_block);
      s_cuda_reduction_tally_block = 0;
      if (cudaerr != cudaSuccess) {
         std::cerr << "\n ERROR in cudaFree call, FILE: "
                   << __FILE__ << " line " << __LINE__ << std::endl;
         exit(1);
       }
   }
}

}  // closing brace for RAJA namespace

#endif  // if defined(RAJA_USE_CUDA)
