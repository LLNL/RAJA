/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing sequential shared memory object type
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

#ifndef RAJA_policy_sequential_shared_memory_HPP
#define RAJA_policy_sequential_shared_memory_HPP


#include <stddef.h>
#include <memory>
#include <vector>
#include "RAJA/config.hpp"
#include "RAJA/pattern/shared_memory.hpp"

namespace RAJA
{


/*!
 * Shared memory, ensures a single copy of data even with thread-private
 * copies of this object.
 *
 * Data is accessible with const capture-by-value copies of this object.
 */
template <typename T, size_t NumElem>
struct SharedMemory<seq_shmem, T, NumElem> : public internal::SharedMemoryBase {
  using self = SharedMemory<seq_shmem, T, NumElem>;
  using element_t = T;

  static constexpr size_t size = NumElem;
  static constexpr size_t num_bytes = NumElem * sizeof(T);

  T *data;
  self const *parent;

  RAJA_INLINE
  constexpr SharedMemory() : data(new T[NumElem]), parent(nullptr) {}

  RAJA_INLINE
  ~SharedMemory()
  {
    if (parent == nullptr) {
      delete[] data;
    }
  }

  RAJA_INLINE
  SharedMemory(self const &c) : data(c.data), parent(&c) {}

  RAJA_INLINE
  RAJA_HOST_DEVICE
  size_t shmem_setup_buffer(size_t) { return num_bytes; }

  template <typename OffsetTuple>
  RAJA_INLINE RAJA_HOST_DEVICE void shmem_set_window(OffsetTuple const &)
  {
  }


  template <typename IDX>
  RAJA_INLINE constexpr T &operator[](IDX i) const
  {
    return data[i];
  }
};


}  // namespace RAJA

#endif
