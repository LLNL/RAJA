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
bool RAJA::detail::shared_memory::setup_enabled = false;


/*!
 * Pointer to a index tuple (for nested::forall) that contains the
 * shared memory window offsets
 */
void *RAJA::detail::shared_memory::window_tuple = nullptr;


/*!
 * Tracks total number of bytes requested for shared memory.
 *
 * This is currently used to tell CUDA how much dynamic shared memory we need
 */
size_t RAJA::detail::shared_memory::total_bytes = 0;
size_t RAJA::detail::shared_memory::window_bytes = 0;

/*! Tracks shared memory objects and their offsets into shared memory
 *
 * The offset is currently only used to point into CUDA dynamic shared memory
 */
std::map<void *, size_t> RAJA::detail::shared_memory::objects;





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
void RAJA::detail::startSharedMemorySetup(void *window_tuple, size_t tuple_size){
  RAJA::detail::shared_memory::setup_enabled = true;
  RAJA::detail::shared_memory::window_tuple = window_tuple;
  RAJA::detail::shared_memory::total_bytes = tuple_size;
  RAJA::detail::shared_memory::window_bytes = tuple_size;
  RAJA::detail::shared_memory::objects.clear();
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
  auto iter = RAJA::detail::shared_memory::objects.find(object);

  // if object is not registered, set aside some more shmem for it
  if(iter == RAJA::detail::shared_memory::objects.end()){

    // Only bother with the registration if we are doing shmem setup
    if(!RAJA::detail::shared_memory::setup_enabled){
      return 0;
    }

    size_t offset = RAJA::detail::shared_memory::total_bytes;
    RAJA::detail::shared_memory::total_bytes += shmem_size;
    RAJA::detail::shared_memory::objects[object] = offset;
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
  RAJA::detail::shared_memory::setup_enabled = false;
}

/*!
 * Returns the size of shared memory required
 *
 * @internal
 *
 */
size_t RAJA::detail::getSharedMemorySize(){
  if(RAJA::detail::shared_memory::window_bytes == RAJA::detail::shared_memory::total_bytes){
    // no shared memory was requested, so don't allocate space for the
    // shmem window
    return 0;
  }
  return RAJA::detail::shared_memory::total_bytes;
}



