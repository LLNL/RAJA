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
// For additional details, please also read RAJA/LICENSE.
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

#include "RAJA/internal/MemUtils_CPU.hpp"

#include "RAJA/util/types.hpp"

#include "RAJA/pattern/reduce.hpp"

#include "RAJA/internal/ThreadUtils_CPU.hpp"

#include <algorithm>
#include <iostream>
#include <string>

#include <stdlib.h>

#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) \
    || defined(__MINGW32__) || defined(__BORLANDC__)
#include <malloc.h>
#endif

namespace RAJA
{

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

void* allocate_aligned(size_t alignment, size_t size)
{
#if defined(HAVE_POSIX_MEMALIGN)
  // posix_memalign available
  void* ret = NULL;
  int err = posix_memalign(&ret, alignment, size);
  return err ? NULL : ret;
#elif defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) \
    || defined(__MINGW32__) || defined(__BORLANDC__)
  // on windows
  return _aligned_malloc(size, alignment);
#else
#error No known aligned allocator available
#endif
}


void free_aligned(void* ptr)
{
#if defined(HAVE_POSIX_MEMALIGN)
  free(ptr);
#elif defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) \
    || defined(__MINGW32__) || defined(__BORLANDC__)
  // on windows
  _aligned_free(ptr);
#else
#error No known aligned allocator available
#endif
}

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
  while (id < RAJA_MAX_REDUCE_VARS && cpu_reduction_id_used[id]) {
    id++;
  }

  if (id >= RAJA_MAX_REDUCE_VARS) {
    std::cerr << "\n Exceeded allowable RAJA CPU reduction count, "
              << "FILE: " << __FILE__ << " line: " << __LINE__ << std::endl;
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
  if (id < RAJA_MAX_REDUCE_VARS) {
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
  int nthreads = getMaxReduceThreadsCPU();

  int block_offset = COHERENCE_BLOCK_SIZE / sizeof(CPUReductionBlockDataType);

  if (s_cpu_reduction_mem_block == 0) {
    int len = nthreads * RAJA_MAX_REDUCE_VARS;
    s_cpu_reduction_mem_block =
        new CPUReductionBlockDataType[len * block_offset];

    atexit(freeCPUReductionMemBlock);
  }

  return &(s_cpu_reduction_mem_block[nthreads * id * block_offset]);
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
  if (s_cpu_reduction_mem_block != 0) {
    delete[] s_cpu_reduction_mem_block;
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
  int nthreads = getMaxReduceThreadsCPU();

  int block_offset = COHERENCE_BLOCK_SIZE / sizeof(Index_type);

  if (s_cpu_reduction_loc_block == 0) {
    int len = nthreads * RAJA_MAX_REDUCE_VARS;
    s_cpu_reduction_loc_block = new Index_type[len * block_offset];

    atexit(freeCPUReductionLocBlock);
  }

  return &(s_cpu_reduction_loc_block[nthreads * id * block_offset]);
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
  if (s_cpu_reduction_loc_block != 0) {
    delete[] s_cpu_reduction_loc_block;
    s_cpu_reduction_loc_block = 0;
  }
}

}  // closing brace for RAJA namespace
