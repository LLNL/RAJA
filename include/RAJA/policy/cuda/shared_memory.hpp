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

#ifdef RAJA_ENABLE_CUDA

namespace RAJA
{

namespace cuda {
namespace detail {

extern bool shared_memory_setup_enabled;
extern size_t shared_memory_total_bytes;

}
}

/*!
 * CUDA shared memory
 */
struct cuda_shmem{};




template<typename T, size_t N>
struct SharedMemory<cuda_shmem, T, N> {
  using Self = SharedMemory<cuda_shmem, T, N>;

  ptrdiff_t offset; // offset into dynamic shared memory, in bytes

  SharedMemory() : offset(0) {}

  SharedMemory(Self const &c) : offset(c.offset){
#ifndef __CUDA_ARCH__
    if(RAJA::cuda::detail::shared_memory_setup_enabled){
      // get current offset into shared memory
      offset = RAJA::cuda::detail::shared_memory_total_bytes;

      // append our memory size
      RAJA::cuda::detail::shared_memory_total_bytes += N*sizeof(T);
    }
#endif
  }


  template<typename IDX>
  RAJA_INLINE
  __device__
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
