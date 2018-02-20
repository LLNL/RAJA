/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing CUDA shared memory object and policy
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

#ifndef RAJA_policy_cuda_shared_memory_HPP
#define RAJA_policy_cuda_shared_memory_HPP


#include "RAJA/config.hpp"
#include "RAJA/pattern/shared_memory.hpp"
#include "RAJA/policy/cuda/policy.hpp"

#ifdef RAJA_ENABLE_CUDA

#include "RAJA/util/defines.hpp"

namespace RAJA
{







/*!
 * Shared Memory object for CUDA kernels.
 *
 * Indexing into this is [0, N), regardless of what block or thread you are.
 *
 * The data is always in CUDA shared memory, so it's block-local.
 */
template<typename T, size_t NumElem>
struct SharedMemory<cuda_shmem, T, NumElem> {
  using self = SharedMemory<cuda_shmem, T, NumElem>;
  using element_t = T;

  static constexpr size_t size = NumElem;
  static constexpr size_t num_bytes = NumElem*sizeof(T);

  long offset; // offset into dynamic shared memory, in bytes
  void *parent;     // pointer to original object

  RAJA_INLINE
  RAJA_HOST_DEVICE
  SharedMemory() :
  offset(-1), parent((void*)this) {}

  RAJA_INLINE
  RAJA_HOST_DEVICE
   SharedMemory(self const &c) :
  offset(c.offset), parent(c.parent)
  {
    // only implement the registration on the HOST
#ifndef __CUDA_ARCH__
    offset = RAJA::detail::registerSharedMemoryObject(parent, NumElem*sizeof(T));
#endif
  }


  template<typename IDX>
  RAJA_INLINE
  RAJA_DEVICE
  T &operator[](IDX i) const {
    // Get the pointer to beginning of dynamic shared memory
    extern __shared__ char my_ptr[];

		// Convert this to a pointer of type T at the beginning of OUR shared mem
    T *T_ptr = reinterpret_cast<T*>((&my_ptr[0]) + offset);

    // Return the i'th element of our buffer
    return T_ptr[i];
  }

};





}  // namespace RAJA

#endif // RAJA_ENABLE_CUDA

#endif
