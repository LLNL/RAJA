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


#include "RAJA/config.hpp"
#include "RAJA/pattern/shared_memory.hpp"
#include <stddef.h>
#include <memory>
#include <vector>

namespace RAJA
{




/*!
 * Shared memory, ensures a single copy of data even with thread-private
 * copies of this object.
 *
 * Data is accessible with const capture-by-value copies of this object.
 */
template<typename T>
struct SharedMemory<seq_shmem, T> {
  using self = SharedMemory<seq_shmem, T>;
  using element_type = T;

  std::shared_ptr<std::vector<T>> data;
  size_t num_elements;

  RAJA_INLINE
  explicit SharedMemory(size_t N) :
    data(std::make_shared<std::vector<T>>(N)),
    num_elements(N)
  {}




  template<typename IDX>
  RAJA_INLINE
  T &operator[](IDX i) const {
    return (*data)[i];
  }
};




}  // namespace RAJA

#endif
