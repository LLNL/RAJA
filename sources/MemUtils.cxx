/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for operations used to manage
 *          memory for ireductiona and other operations.
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#include "RAJA/MemUtils.hxx"

#include "RAJA/int_datatypes.hxx"

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(RAJA_USE_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include<string>
#include<iostream>

namespace RAJA {

//////////////////////////////////////////////////////////////////////
//
// Utilities and methods for CPU reductions.
//
//////////////////////////////////////////////////////////////////////

//
// Static array used to keep track of which unique ids
// for CUDA reduction objects are used and which are not.
//
static bool cpu_reduction_id_used[RAJA_MAX_REDUCE_VARS];

// 
// Pointer to hold shared memory block for RAJA-CPU reductions.
//
CPUReductionBlockDataType* s_cpu_reduction_mem_block = 0;

//
// Pointer to hold shared memory block for index locations in RAJA-CPU 
// "loc" reductions.
//
Index_type* s_cpu_reduction_loc_block = 0;


/*
*************************************************************************
*
* Return available valid reduction id and record reduction type for that
* id, or complain and exit if no ids are available.
*
*************************************************************************
*/
int getCPUReductionId()
{
   static int first_time_called = true;

   if (first_time_called) {

      for (int id = 0; id < RAJA_MAX_REDUCE_VARS; ++id) {
         cpu_reduction_id_used[id] = false;
      }

      first_time_called = false;
   }

   int id = 0;
   while ( id < RAJA_MAX_REDUCE_VARS && cpu_reduction_id_used[id] ) {
     id++;
   }

   if ( id >= RAJA_MAX_REDUCE_VARS ) {
      std::cerr << "\n Exceeded allowable RAJA CPU reduction count, "
                << "FILE: "<< __FILE__ << " line: "<< __LINE__ << std::endl;
      exit(1);
   }

   cpu_reduction_id_used[id] = true;

   return id;
}

/*
*************************************************************************
*
* Release given redution id and make inactive.  
*
*************************************************************************
*/
void releaseCPUReductionId(int id)
{
   if ( id < RAJA_MAX_REDUCE_VARS ) {
      cpu_reduction_id_used[id] = false;
   }
}

/*
*************************************************************************
*
* Return pointer into shared RAJA-CPU reduction memory block for 
* reduction object with given id. Allocates block if not alreay allocated. 
*
*************************************************************************
*/
CPUReductionBlockDataType* getCPUReductionMemBlock(int id)
{
   int nthreads = 1;
#if defined(_OPENMP)
   nthreads = omp_get_max_threads();
#endif

   int block_offset = COHERENCE_BLOCK_SIZE/sizeof(CPUReductionBlockDataType);

   if (s_cpu_reduction_mem_block == 0) {
      int len = nthreads * RAJA_MAX_REDUCE_VARS;
      s_cpu_reduction_mem_block = 
         new CPUReductionBlockDataType[len*block_offset];

      atexit(freeCPUReductionMemBlock);
   }

   return &(s_cpu_reduction_mem_block[nthreads * id * block_offset]) ;
}


/*
*************************************************************************
*
* Free managed memory block used in RAJA-CPU reductions.
*
*************************************************************************
*/
void freeCPUReductionMemBlock()
{
   if ( s_cpu_reduction_mem_block != 0 ) {
      delete [] s_cpu_reduction_mem_block;
      s_cpu_reduction_mem_block = 0; 
   }
}

/*
*************************************************************************
*
* Return pointer into shared RAJA-CPU memory block index location for
* reduction object with given id. Allocates block if not alreay allocated.
*
*************************************************************************
*/
Index_type* getCPUReductionLocBlock(int id)
{
   int nthreads = 1;
#if defined(_OPENMP)
   nthreads = omp_get_max_threads();
#endif

   int block_offset = COHERENCE_BLOCK_SIZE/sizeof(Index_type);

   if (s_cpu_reduction_loc_block == 0) {
      int len = nthreads * RAJA_MAX_REDUCE_VARS;
      s_cpu_reduction_loc_block =
         new Index_type[len*block_offset];

      atexit(freeCPUReductionLocBlock);
   }

   return &(s_cpu_reduction_loc_block[nthreads * id * block_offset]) ;
}


/*
*************************************************************************
*
* Free managed index location memory block used in RAJA-CPU reductions.
*
*************************************************************************
*/
void freeCPUReductionLocBlock()
{
   if ( s_cpu_reduction_loc_block != 0 ) {
      delete [] s_cpu_reduction_loc_block;
      s_cpu_reduction_loc_block = 0;
   }
}



#if defined(RAJA_USE_CUDA)

//////////////////////////////////////////////////////////////////////
//
// Utilities and methods for CUDA reductions.
//
//////////////////////////////////////////////////////////////////////


//
// Static array used to keep track of which unique ids 
// for CUDA reduction objects are used and which are not.
//
static bool cuda_reduction_id_used[RAJA_MAX_REDUCE_VARS];

//
// Used to track current CUDA grid size used in forall methods, so
// reduction objects can properly finalize reductions in their accesor
// methods
//
size_t s_current_grid_size = 0;

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
* Set current CUDA grid size used in forall methods as given arg value
* so it can be used in other methods (i.e., reduction finalization).
*
*************************************************************************
*/
void setCurrentGridSize(size_t s)
{
   s_current_grid_size = s;
}

/*
*************************************************************************
*
* Retrieve current CUDA grid size value.
*
*************************************************************************
*/
size_t getCurrentGridSize()
{
   return s_current_grid_size;
}

/*
*************************************************************************
*
* Return pointer to shared RAJA-CUDA managed reduction memory block.
* Allocates block if not alreay allocated.
*
*************************************************************************
*/
CudaReductionBlockDataType* getCudaReductionMemBlock()
{
   if (s_cuda_reduction_mem_block == 0) {
      int len = RAJA_CUDA_REDUCE_BLOCK_LENGTH * RAJA_MAX_REDUCE_VARS +
                                                RAJA_MAX_REDUCE_VARS;
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

   return s_cuda_reduction_mem_block;
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
* Return offset into shared RAJA-Cuda reduction memory block for
* reduction object with given id.
*
*************************************************************************
*/
int getCudaReductionMemBlockOffset(int id)
{
   return (id * RAJA_CUDA_REDUCE_BLOCK_LENGTH + id);
}

#endif // #if defined(RAJA_USE_CUDA)



}  // closing brace for RAJA namespace
