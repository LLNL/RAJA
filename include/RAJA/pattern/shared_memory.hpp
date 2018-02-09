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


#include "RAJA/config.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/util/types.hpp"
#include "camp/camp.hpp"
#include <stddef.h>
#include <memory>
#include <vector>

namespace RAJA
{



/*!
 * Creates a shared memory object with elements of type T.
 * The Policy determines
 */
template<typename SharedPolicy, typename T, size_t NumElem>
struct SharedMemory;



namespace detail {
  void startSharedMemorySetup(void *window_tuple = nullptr, size_t tuple_size = 0);

  void *getSharedMemoryWindow();

  size_t registerSharedMemoryObject(void *object, size_t shmem_size);

  void finishSharedMemorySetup();

  size_t getSharedMemorySize();

}



}  // namespace RAJA

#endif
