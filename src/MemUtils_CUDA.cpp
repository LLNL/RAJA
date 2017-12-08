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
// Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/config.hpp"

#include "RAJA/policy/cuda/shared_memory.hpp"

#include <algorithm>
#include <map>

#if defined(RAJA_ENABLE_CUDA)

#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"

#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"



namespace RAJA
{

namespace cuda
{

namespace detail
{
//
/////////////////////////////////////////////////////////////////////////////
//
// Variables representing the state of execution.
//
/////////////////////////////////////////////////////////////////////////////
//

//! State of the host code globally
cudaInfo g_status;

//! State of the host code in this thread
cudaInfo tl_status;
#if defined(RAJA_ENABLE_OPENMP) && defined(_OPENMP)
#pragma omp threadprivate(tl_status)
#endif

//! State of raja cuda stream synchronization for cuda reducer objects
std::unordered_map<cudaStream_t, bool> g_stream_info_map{ {cudaStream_t(0), true} };



//
/////////////////////////////////////////////////////////////////////////////
//
// Variables for tracking dynamic shared memory
//
/////////////////////////////////////////////////////////////////////////////
//

//! Activates shared memory configuration in SharedMemory<cuda_shmem> objects
bool shared_memory_setup_enabled = false;

//! Tracks total number of bytes requested for shared memory
size_t shared_memory_total_bytes = 0;


}  // closing brace for detail namespace

}  // closing brace for cuda namespace

}  // closing brace for RAJA namespace


#endif  // if defined(RAJA_ENABLE_CUDA)


std::map<void *, size_t> shared_memory_objects;

/*!
 * Marks start of shared memory setup
 *
 * @internal
 *
 * The loop body should be copied by value, exactly once, between a call to
 * this function and finishSharedMemorySetup
 *
 */
void RAJA::detail::startSharedMemorySetup(){
#ifdef RAJA_ENABLE_CUDA
  RAJA::cuda::detail::shared_memory_setup_enabled = true;
  RAJA::cuda::detail::shared_memory_total_bytes = 0;
#endif
  shared_memory_objects.clear();
}

/*!
 * Registers a shared memory object, and it's size requirement
 *
 *
 *
 * @param object
 * @param shmem_size
 */
size_t RAJA::detail::registerSharedMemoryObject(void *object, size_t shmem_size){
  // look up object
  auto iter = shared_memory_objects.find(object);

  // if object is not registered, set aside some more shmem for it
  if(iter == shared_memory_objects.end()){
    size_t offset = RAJA::cuda::detail::shared_memory_total_bytes;
    RAJA::cuda::detail::shared_memory_total_bytes += shmem_size;
    shared_memory_objects[object] = offset;
    return offset;
  }

  // if it's already registered, just return its existing offset
  return iter->second;
}

/*!
 * Marks end of shared memory setup
 *
 * @internal
 *
 * At this point, we should have calculated how much shared memory to allocate
 * for CUDA, or other backend.
 */
void RAJA::detail::finishSharedMemorySetup(){
#ifdef RAJA_ENABLE_CUDA
  RAJA::cuda::detail::shared_memory_setup_enabled = false;
#endif
}

/*!
 * Returns the size of shared memory required
 *
 * @internal
 *
 */
size_t RAJA::detail::getSharedMemorySize(){
  return RAJA::cuda::detail::shared_memory_total_bytes;
}



