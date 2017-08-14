/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file defining prototypes for routines used to manage
 *          memory for CPU reductions and other operations.
 *
 ******************************************************************************
 */

#ifndef RAJA_MemUtils_CPU_HPP
#define RAJA_MemUtils_CPU_HPP

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

#include "RAJA/config.hpp"

#include "RAJA/util/types.hpp"

#include <cstddef>

namespace RAJA
{

///
/// Portable aligned memory allocation
///
void* allocate_aligned(size_t alignment, size_t size);

///
/// Portable aligned memory allocation
///
template <typename T>
T* allocate_aligned_type(size_t alignment, size_t size)
{
  return reinterpret_cast<T*>(allocate_aligned(alignment, size));
}


///
/// Portable aligned memory free - required for Windows
///
void free_aligned(void* ptr);

///
/// Typedef defining common data type for RAJA-CPU reduction data blocks
/// (use this in all cases to avoid type confusion).
///
using CPUReductionBlockDataType = double;

/*!
*************************************************************************
*
* Return available valid reduction id and record reduction type for
* that id, or complain and exit if no ids are available.
*
*************************************************************************
*/
int getCPUReductionId();

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
 *         data for reduction object with given id.
 *
 *         Allocates data block if it isn't allocated already.
 *
 * NOTE: Block size will be of one of the following sizes:
 *
 *       When compiled with OpenMP :
 *
 *          omp_get_max_threads() * MAX_REDUCE_VARS_CPU *
 *          COHERENCE_BLOCK_SIZE/sizeof(CPUReductionBlockDataType)
 *
 *       When compiled without OpenMP :
 *
 *          MAX_REDUCE_VARS_CPU *
 *          COHERENCE_BLOCK_SIZE/sizeof(CPUReductionBlockDataType)
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
 * \brief  Return pointer into shared memory block for index location in
 *         RAJA-CPU "loc" reductions for reduction object with given id.
 *
 *         Allocates data block if it isn't allocated already.
 *
 * NOTE: Block size will be of one of the following sizes:
 *
 *       When compiled with OpenMP :
 *
 *          omp_get_max_threads() * MAX_REDUCE_VARS_CPU *
 *          COHERENCE_BLOCK_SIZE/sizeof(Index_type)
 *
 *       When compiled without OpenMP :
 *
 *          MAX_REDUCE_VARS_CPU *
 *          COHERENCE_BLOCK_SIZE/sizeof(Index_type)
 *
 ******************************************************************************
 */
Index_type* getCPUReductionLocBlock(int id);

/*!
 ******************************************************************************
 *
 * \brief  Free managed memory location index block used in RAJA-CPU reductions.
 *
 ******************************************************************************
 */
void freeCPUReductionLocBlock();

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
