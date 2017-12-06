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
#include <stddef.h>

namespace RAJA
{



/*!
 * Creates a shared memory object with N elements of type T.
 * The Policy determines
 */
template<typename SharedPolicy, typename T, size_t N>
struct SharedMemory {
};





/*!
 * Not-really-shared-memory, just creates a fixed length buffer on the stack
 */
struct seq_shmem{};


template<typename T, size_t N>
struct SharedMemory<seq_shmem, T, N> {

  // TODO: should use alignment, etc.?!?!
  T data[N];


  template<typename IDX>
  RAJA_INLINE
  T &operator[](IDX i) const {
    return data[i];
  }
};


namespace detail {
  void startSharedMemorySetup();
  void finishSharedMemorySetup();

}



}  // namespace RAJA

#endif
