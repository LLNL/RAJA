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

#include "RAJA/pattern/shared_memory.hpp"

#include <algorithm>
#include <map>



//
/////////////////////////////////////////////////////////////////////////////
//
// Variables for tracking dynamic shared memory
//
/////////////////////////////////////////////////////////////////////////////
//

/*!
 * Identifies if shared memory configuration is active.
 *
 * This is controlled with the startSharedMemorySetup and
 * finishSharedMemorySetup functions
 */
static bool shared_memory_setup_enabled = false;

/*!
 * Tracks total number of bytes requested for shared memory.
 *
 * This is currently used to tell CUDA how much dynamic shared memory we need
 */
static size_t shared_memory_total_bytes = 0;

/*! Tracks shared memory objects and their offsets into shared memory
 *
 * The offset is currently only used to point into CUDA dynamic shared memory
 */
static std::map<void *, size_t> shared_memory_objects;





//
/////////////////////////////////////////////////////////////////////////////
//
// Functions for tracking dynamic shared memory
//
/////////////////////////////////////////////////////////////////////////////
//





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
  shared_memory_setup_enabled = true;
  shared_memory_total_bytes = 0;
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

    // Only bother with the registration if we are doing shmem setup
    if(!shared_memory_setup_enabled){
      return 0;
    }

    size_t offset = shared_memory_total_bytes;
    shared_memory_total_bytes += shmem_size;
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
  shared_memory_setup_enabled = false;
}

/*!
 * Returns the size of shared memory required
 *
 * @internal
 *
 */
size_t RAJA::detail::getSharedMemorySize(){
  return shared_memory_total_bytes;
}



