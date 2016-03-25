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
 * \brief   Implementation file for routines used to manage
 *          memory for CPU reductions and other operations.
 *
 ******************************************************************************
 */

#include "RAJA/MemUtils_CPU.hxx"

#include "RAJA/int_datatypes.hxx"

#include "RAJA/reducers.hxx"

#include "RAJA/ThreadUtils_CPU.hxx"


#include<algorithm>
#include<string>
#include<iostream>

namespace RAJA {

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
   int nthreads = getMaxThreadsCPU();

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
   int nthreads = getMaxThreadsCPU();

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


}  // closing brace for RAJA namespace
