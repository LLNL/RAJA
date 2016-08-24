/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file defining prototypes for routines used to manage
 *          memory for CUDA reductions and other operations.
 *
 ******************************************************************************
 */

#ifndef RAJA_MemUtils_CUDA_HXX
#define RAJA_MemUtils_CUDA_HXX

#include "RAJA/config.hxx"

#if defined(RAJA_ENABLE_CUDA)

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

#include "RAJA/int_datatypes.hxx"

namespace RAJA
{
/*!
 * \def RAJA_CUDA_MAX_NUM_BLOCKS
 * Maximum number of blocks that RAJA will launch
 */ 
#define RAJA_CUDA_MAX_NUM_BLOCKS (1024 * 16)

/*!
 * \def RAJA_CUDA_REDUCE_BLOCK_LENGTH
 * Size of reduction memory block for each reducer object (value based on
 * rough estimate of "worst case" maximum numvber of blocks)
 */
#define RAJA_CUDA_REDUCE_BLOCK_LENGTH RAJA_CUDA_MAX_NUM_BLOCKS

/*!
 * \def RAJA_CUDA_REDUCE_TALLY_LENGTH
 * Reduction Tallies are computed into a small block to minimize memory motion
 * Set to Max Number of Reduction Variables
 */
#define RAJA_CUDA_REDUCE_TALLY_LENGTH RAJA_MAX_REDUCE_VARS

/*!
 * \def RAJA_CUDA_REDUCE_VAR_MAXSIZE
 * Used in CudaReductionDummyDataType to allocate arrays and accommodate all the Reduction types
 * includes the size of the index variable for Loc reductions
 */
#define RAJA_CUDA_REDUCE_VAR_MAXSIZE 16

/*!
 * \def MACROSTR(x)
 * Used in the static_assert macros
 */
#define STR(x) #x
#define MACROSTR(x) STR(x)

/*!
 * \def RAJA_STRUCT_ALIGNAS
 * abstracts the alignas keyword
 */
#define RAJA_STRUCT_ALIGNAS(dalign) alignas(dalign)

/*!
 * \brief Type used to keep track of the grid size on the device
 */
typedef unsigned int GridSizeType;

/*!
 ******************************************************************************
 *
 * \brief Types used to simplify memory allocation for cuda reductions.
 *
 * These dummy types are designed to fit types of size 16 bytes or less.
 *
 ******************************************************************************
 */
struct RAJA_STRUCT_ALIGNAS(RAJA_CUDA_REDUCE_VAR_MAXSIZE) CudaReductionDummyDataType {
  unsigned char data[RAJA_CUDA_REDUCE_VAR_MAXSIZE];
};
struct RAJA_STRUCT_ALIGNAS(DATA_ALIGN) CudaReductionDummyBlockType {
  CudaReductionDummyDataType values[RAJA_CUDA_REDUCE_BLOCK_LENGTH];
};
struct CudaReductionDummyTallyType {
  CudaReductionDummyDataType dummy_val;
  GridSizeType dummy_retiredBlocks;
};

/*!
 ******************************************************************************
 *
 * \brief Types used to simplify typed memory use in reductions.
 *
 * These types fit within the dummy types, and that is checked in static 
 * asserts in the reduction classes.
 *
 ******************************************************************************
 */
template <typename T>
struct CudaReductionBlockType {
  T values[RAJA_CUDA_REDUCE_BLOCK_LENGTH];
};
template <typename T>
struct CudaReductionLocBlockType {
  T values[RAJA_CUDA_REDUCE_BLOCK_LENGTH];
  Index_type indices[RAJA_CUDA_REDUCE_BLOCK_LENGTH];
};

template  <typename T>
struct CudaReductionLocType {
  T val;
  Index_type idx;
};

/*!
 ******************************************************************************
 *
 * \brief Types used in the device tally and host tally cache to hold reduced
 *        values.
 *
 ******************************************************************************
 */
template <typename T>
struct CudaReductionTallyType {
  T tally;
  GridSizeType retiredBlocks;
};
template <typename T>
struct CudaReductionTallyTypeAtomic {
  T tally;
};
template <typename T>
struct CudaReductionLocTallyType {
  CudaReductionLocType<T> tally;
  GridSizeType retiredBlocks;
};


/*!
 ******************************************************************************
 *
 * \brief Get a valid reduction id, or complain and exit if no valid id is 
 *        available.
 *
 * \return int the next available valid reduction id.
 *
 ******************************************************************************
 */
int getCudaReductionId();

/*!
 ******************************************************************************
 *
 * \brief Release given reduction id and make inactive.
 *
 ******************************************************************************
 */
void releaseCudaReductionId(int id);

/*!
 ******************************************************************************
 *
 * \brief Get tally block for reducer object with given id.
 *
 * \param[out] host_tally pointer to host tally cache slot.
 * \param[out] device_tally pointer to device tally slot.
 *
 * NOTE: Tally Block size will be:
 *
 *          sizeof(CudaReductionDummyTallyType) * RAJA_MAX_REDUCE_VARS
 *
 *       For each reducer object, we want a chunk of device memory that
 *       holds the reduced value and a small number of anciliary variables.
 *
 ******************************************************************************
 */
void getCudaReductionTallyBlock(int id, void** host_tally, void** device_tally);

/*!
 ******************************************************************************
 *
 * \brief Release tally block for reducer object with given id.
 * 
 ******************************************************************************
 */
void releaseCudaReductionTallyBlock(int id);

/*!
 ******************************************************************************
 *
 * \brief Sets up state variales before the loop body is copied and the kernel
 *        is launched.
 *
 ******************************************************************************
 */
void beforeCudaKernelLaunch();

/*!
 ******************************************************************************
 *
 * \brief Resets state variables after kernel launch.
 *
 ******************************************************************************
 */ 
void afterCudaKernelLaunch();

/*!
 ******************************************************************************
 *
 * \brief Updates host tally cache for read by reduction varaible with id.
 *
 ******************************************************************************
 */
void beforeCudaReadTallyBlockAsync(int id);

void beforeCudaReadTallyBlock(int id);

/*!
 ******************************************************************************
 *
 * \brief  Earmark amount of device shared memory and get byte offset into
 *         device shared memory.
 *
 * \return int Byte offset into dynamic shared memory.
 *
 * /param[in] num_threads number of threads expected by this reduction variable.
 * \param[in] size Size of shared memory in bytes needed for each thread.
 *
 ******************************************************************************
 */
int getCudaSharedmemOffset(int id, int num_threads, int size);

/*!
 ******************************************************************************
 *
 * \brief  Get the amount in bytes of shared memory required for the current
 *         kernel launch and check reduction policy num_threads match launch 
 *         num_threads.
 *
 * \param[in] num_threads number of threads launched in this kernel invocation.
 *
 ******************************************************************************
 */
int getCudaSharedmemAmount(int launch_num_threads);

/*!
 ******************************************************************************
 *
 * \brief  Free managed memory block used in RAJA-Cuda reductions.
 *
 ******************************************************************************
 */

void freeCudaReductionTallyBlock();

/*!
 ******************************************************************************
 *
 * \brief  Get device memory block for RAJA-CUDA reduction variable  with
 *         given id.
 *
 *         Allocates data block if it isn't allocated already.
 *
 * \param[out] device_memblock Pointer to device memory block.
 *
 * NOTE: Total Block size will be:
 *
 *          sizeof(CudaReductionDummyDataType) *
 *            RAJA_MAX_REDUCE_VARS * RAJA_CUDA_REDUCE_BLOCK_LENGTH
 *
 *       For each reducer object, we want a chunk of device memory that
 *       holds RAJA_CUDA_REDUCE_BLOCK_LENGTH slots for the reduction
 *       value for each thread block.
 *
 ******************************************************************************
 */

void getCudaReductionMemBlock(int id, void** device_memblock);

/*!
 ******************************************************************************
 *
 * \brief  Free device memory blocks used in RAJA-Cuda reductions.
 *
 ******************************************************************************
 */
void freeCudaReductionMemBlock();

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_CUDA

#endif  // closing endif for header file include guard
