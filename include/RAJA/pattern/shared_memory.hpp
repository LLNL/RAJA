/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing shared memory object type
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

#ifndef RAJA_pattern_shared_memory_HPP
#define RAJA_pattern_shared_memory_HPP


#include <stddef.h>
#include <map>
#include <memory>
#include <vector>
#include "RAJA/config.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/util/types.hpp"
#include "camp/camp.hpp"

namespace RAJA
{


/*!
 * Creates a shared memory object with elements of type T.
 * The Policy determines
 */
template <typename SharedPolicy, typename T, size_t NumElem>
struct SharedMemory;


namespace detail
{

struct shared_memory {
  /*!
   * Identifies if shared memory configuration is active.
   *
   * This is controlled with the startSharedMemorySetup and
   * finishSharedMemorySetup functions
   */
  static bool setup_enabled;


  /*!
   * Pointer to a index tuple (for nested::forall) that contains the
   * shared memory window offsets
   */
  static void *window_tuple;


  /*!
   * Tracks total number of bytes requested for shared memory.
   *
   * This is currently used to tell CUDA how much dynamic shared memory we need
   */
  static size_t total_bytes;
  static size_t window_bytes;

  /*! Tracks shared memory objects and their offsets into shared memory
   *
   * The offset is currently only used to point into CUDA dynamic shared memory
   */
  static std::map<void *, size_t> objects;
};

void startSharedMemorySetup(void *window_tuple = nullptr,
                            size_t tuple_size = 0);

RAJA_INLINE
void *getSharedMemoryWindow()
{
  return RAJA::detail::shared_memory::window_tuple;
}

size_t registerSharedMemoryObject(void *object, size_t shmem_size);

void finishSharedMemorySetup();

size_t getSharedMemorySize();
}


}  // namespace RAJA

#endif
