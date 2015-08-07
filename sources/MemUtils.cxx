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

#if 0 // RDH Will we ever need something like this?
//
// Array holding rection types for valid reduction ids.
//
static ReductionType cpu_reduction_type[RAJA_MAX_REDUCE_VARS];
#else
int s_cpu_reduction_id = -1;
#endif

// 
// Pointers to hold shared memory blocks for RAJA-CPU reductions.
//
CPUReductionBlockDataType* s_cpu_reduction_mem_block = 0;
int s_block_offset = 0;

CPUReductionBlockDataType* s_cpu_reduction_init_vals = 0;


/*
*************************************************************************
*
* Return available valid reduction id and record reduction type for that
* id, or complain and exit if no ids are available.
*
*************************************************************************
*/
#if 0 // RDH Will we ever need something like this?
int getCPUReductionId(ReductionType type)
{
   static int first_time_called = true;

   if (first_time_called) {

      for (int id = 0; id < RAJA_MAX_REDUCE_VARS; ++id) {
         cpu_reduction_type[id] = _INACTIVE_;
      }

      first_time_called = false;
   }

   int id = 0;
   while ( id < RAJA_MAX_REDUCE_VARS && 
           cpu_reduction_type[id] != _INACTIVE_ ) {
     id++;    
   }

   if ( id >= RAJA_MAX_REDUCE_VARS ) {
      std::cerr << "\n Exceeded allowable RAJA reduction count, "
                << "FILE: "<< __FILE__ << " line: "<< __LINE__ << std::endl;
      exit(1);
   } else {
      cpu_reduction_type[id] = type;
   }

   return id;
}
#else
int getCPUReductionId()
{
   s_cpu_reduction_id++;

   if ( s_cpu_reduction_id >= RAJA_MAX_REDUCE_VARS ) {
      std::cerr << "\n Exceeded allowable RAJA reduction count, "
                << "FILE: "<< __FILE__ << " line: "<< __LINE__ << std::endl;
      exit(1);
   }

   return s_cpu_reduction_id;
}
#endif

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
      s_cpu_reduction_id--;
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
   s_block_offset = nthreads;

   if (s_cpu_reduction_mem_block == 0) {
      int len = nthreads * RAJA_MAX_REDUCE_VARS;
      s_cpu_reduction_mem_block = new CPUReductionBlockDataType[len];

      atexit(freeCPUReductionMemBlock);
   }

   return &(s_cpu_reduction_mem_block[s_block_offset * id]) ;
}


/*
*************************************************************************
*
* Free managed memory blocks used in RAJA-CPU reductions.
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
* Set value in shared memory block that holds initial values for RAJA-CPU 
* reductions. Allocates block if not already allocated.
*
*************************************************************************
*/
void setCPUReductionInitValue(int id, CPUReductionBlockDataType val)
{
   if (s_cpu_reduction_init_vals == 0) {
      int len = RAJA_MAX_REDUCE_VARS;
      s_cpu_reduction_init_vals = new CPUReductionBlockDataType[len];

      atexit(freeCPUReductionInitData);
   }
   
   s_cpu_reduction_init_vals[id] = val;
}

/*
*************************************************************************
*
* Get value in shared memory block that holds initial values for RAJA-CPU
* reductions.
*
*************************************************************************
*/
CPUReductionBlockDataType getCPUReductionInitValue(int id)
{
   return s_cpu_reduction_init_vals[id];
}


/*
*************************************************************************
*
* Free managed memory block used to hold initial values for RAJA-CPU 
* reductions.
*
*************************************************************************
*/
void freeCPUReductionInitData()
{
   if ( s_cpu_reduction_init_vals != 0 ) {
      delete [] s_cpu_reduction_init_vals;
      s_cpu_reduction_init_vals = 0;
   }
}


#if defined(RAJA_USE_CUDA)

//////////////////////////////////////////////////////////////////////
//
// Utilities and methods for CUDA reductions.
//
//////////////////////////////////////////////////////////////////////


//
// Counter variable for number of active reducer objects, used to
// computer offset into shared managed reduction memory block.
// Must be accessable from both host and device, so allocated as
// a CUDA managed array.
//
int* s_gid = 0;

//
// Used to track current CUDA grid size used in forall methods, so
// reduction objects can properly finalize reductions in their accesor
// methods
//
size_t s_current_grid_size = 0;

//
// Pointers to hold shared memory blocks for RAJA-Cuda reductions.
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
int* getCudaReductionId()
{
  if (s_gid == 0) {
      cudaError_t cudaerr = cudaMallocManaged((void **)&s_gid, 1,
                                              cudaMemAttachGlobal);

      if ( cudaerr != cudaSuccess ) {
         std::cerr << "\n ERROR in cudaMallocManaged call, FILE: "
                   << __FILE__ << " line " << __LINE__ << std::endl;
         exit(1);
      }
      s_gid[0] = -1;

      atexit(freeCudaReductionIdMem);
   }

   s_gid[0] += 1;

   if ( s_gid[0] >= RAJA_MAX_REDUCE_VARS ) {
      std::cerr << "\n Exceeded allowable RAJA CUDA reduction object count, "
                << "FILE: "<< __FILE__ << " line: "<< __LINE__ << std::endl;
      exit(1);
   }

   return s_gid;
}

/*
*************************************************************************
*
* Free managed memory for RAJA-Cuda reduction ids.
*
*************************************************************************
*/
void freeCudaReductionIdMem()
{
   if ( s_gid != 0 ) {
      cudaError_t cudaerr = cudaFree(s_gid);
      s_gid = 0;
      if (cudaerr != cudaSuccess) {
         std::cerr << "\n ERROR in cudaFree call, FILE: "
                      << __FILE__ << " line " << __LINE__ << std::endl;
         exit(1);
       }
   }
}

#if 0 // RDH We can't use this b/c we can't access managed data pointer
      // that lives on host from device methods (i.e., reduction object
      // destructors).
/*
*************************************************************************
*
* Release given redution id and make inactive.
*
*************************************************************************
*/
void releaseCudaReductionId(int id)
{
   if (!s_gid == 0) {
      s_gid[0] -= 1;
   }
}
#endif

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
* Return pointer to shared RAJA-Cuda reduction memory block.
* Allocates block if not alreay allocated.
*
*************************************************************************
*/
CudaReductionBlockDataType* getCudaReductionMemBlock()
{
   if (s_cuda_reduction_mem_block == 0) {
      int len = RAJA_CUDA_REDUCE_BLOCK_LENGTH * RAJA_MAX_REDUCE_VARS;
      cudaError_t cudaerr = cudaMallocManaged((void **)&s_cuda_reduction_mem_block,
                            sizeof(CudaReductionBlockDataType)*len,
                            cudaMemAttachGlobal);

      if ( cudaerr != cudaSuccess ) {
         std::cerr << "\n ERROR in cudaMallocManaged call, FILE: "
                   << __FILE__ << " line " << __LINE__ << std::endl;
         exit(1);
      }

      atexit(freeCudaReductionMemBlock);
   }

   return s_cuda_reduction_mem_block;
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
   return (id * RAJA_CUDA_REDUCE_BLOCK_LENGTH);
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

#endif // #if defined(RAJA_USE_CUDA)



}  // closing brace for RAJA namespace
