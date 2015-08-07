/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for operations used to manage
 *          memory for CPU operations.
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#include "RAJA/MemUtilsCPU.hxx"

#include "RAJA/int_datatypes.hxx"

#if defined(_OPENMP)
#include <omp.h>
#endif

#include<string>
#include<iostream>

namespace RAJA {

//
// Array holding rection types for valid reduction ids.
//
static ReductionType cpu_reduction_type[RAJA_MAX_REDUCE_VARS];

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
      cpu_reduction_type[id] = _INACTIVE_;
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
   }

   atexit(freeCPUReductionMemBlock);

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
   }
   
   s_cpu_reduction_init_vals[id] = val;

   atexit(freeCPUReductionInitData);
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

}  // closing brace for RAJA namespace
