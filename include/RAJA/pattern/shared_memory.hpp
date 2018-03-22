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
#include "RAJA/config.hpp"
#include "RAJA/internal/LegacyCompatibility.hpp"
#include "RAJA/util/Operators.hpp"
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


namespace internal
{


/*!
 * Shared memory objects that utilize the ShmemWindow should be derived from
 * this type to enable run-time functions like shmem_set_window, etc.
 */
struct SharedMemoryBase {
};


/**
 * Sets the shared memory window for any objects that are
 * aware of shared memory windows.
 *
 * This specialization allows for NOPs for non-shared memory objects.
 */
template <typename T>
RAJA_HOST_DEVICE RAJA_INLINE typename std::
    enable_if<!std::is_base_of<SharedMemoryBase, camp::decay<T>>::value,
              size_t>::type
    shmem_setup_buffer(T &, size_t)
{
  return 0;
}


/**
 * Configures the shared memory buffer for objects that use shared memory.
 *
 * These objects must derive from ShmemWindowBase.
 *
 * @param shmem  The shared memory object in the parameter tuple.
 * @param offset The current number of bytes already used by shared memory.
 * @return The number of bytes this shared memory object consumes.
 */
template <typename T>
RAJA_HOST_DEVICE RAJA_INLINE typename std::
    enable_if<std::is_base_of<SharedMemoryBase, camp::decay<T>>::value,
              size_t>::type
    shmem_setup_buffer(T &shmem, size_t offset)
{
  return shmem.shmem_setup_buffer(offset);
}


/**
 * Configures the shared memory buffer for objects that use shared memory.
 * *
 * This specialization allows for NOPs for non-shared memory objects.
 */
template <typename T, typename OffsetTuple>
RAJA_HOST_DEVICE RAJA_INLINE typename std::
    enable_if<!std::is_base_of<SharedMemoryBase, camp::decay<T>>::value>::type
    shmem_set_window(T &, OffsetTuple const &)
{
}


/**
 * Sets the shared memory window for any objects that are
 * aware of shared memory windows.
 *
 * These objects must derive from ShmemWindowBase.
 */
template <typename T, typename OffsetTuple>
RAJA_HOST_DEVICE RAJA_INLINE typename std::
    enable_if<std::is_base_of<SharedMemoryBase, camp::decay<T>>::value>::type
    shmem_set_window(T &shmem, OffsetTuple const &offset_tuple)
{
  shmem.shmem_set_window(offset_tuple);
}


/*!
 * Helper that walks through the parameter tuple, and calls shared memory
 * operations on each one.
 */
template <camp::idx_t Idx>
struct ShmemHelper {

  template <typename ParamTuple>
  static RAJA_INLINE size_t setup_buffers(ParamTuple &param_tuple)
  {
    size_t offset = ShmemHelper<Idx - 1>::setup_buffers(param_tuple);

    // assign shmem offset, and compute offset of next buffer
    size_t new_offset =
        offset + shmem_setup_buffer(camp::get<Idx>(param_tuple), offset);

    return new_offset;
  }

  template <typename ParamTuple, typename OffsetTuple>
  static RAJA_INLINE RAJA_HOST_DEVICE void set_window(
      ParamTuple &param_tuple,
      OffsetTuple const &offset_tuple)
  {
    shmem_set_window(camp::get<Idx>(param_tuple), offset_tuple);

    ShmemHelper<Idx - 1>::set_window(param_tuple, offset_tuple);
  }
};

/*
 * Shared memory Helper terminator.
 */
template <>
struct ShmemHelper<-1> {

  template <typename ParamTuple>
  static RAJA_INLINE size_t setup_buffers(ParamTuple &)
  {
    return 0;
  }

  template <typename ParamTuple, typename OffsetTuple>
  static RAJA_INLINE RAJA_HOST_DEVICE void set_window(ParamTuple &,
                                                      OffsetTuple const &)
  {
  }
};


/*!
 * Setup shared memory buffers for shared memory objects that have been passed
 * into RAJA kernels via the parameter tuple.
 *
 * @return Total bytes of shared memory needed
 */
template <typename... Params>
RAJA_INLINE size_t shmem_setup_buffers(camp::tuple<Params...> &param_tuple)
{

  return ShmemHelper<((camp::idx_t)sizeof...(Params)) - 1>::setup_buffers(
      param_tuple);
}


/*!
 * Updates shared memory window offsets for shared memory objects that have
 * been passed into RAJA kernels via the parameter tuple.
 */
template <typename... Params, typename OffsetTuple>
RAJA_HOST_DEVICE RAJA_INLINE void shmem_set_windows(
    camp::tuple<Params...> &param_tuple,
    OffsetTuple const &offset_tuple)
{

  ShmemHelper<((camp::idx_t)sizeof...(Params)) - 1>::set_window(param_tuple,
                                                                offset_tuple);
}


}  // namespace internal

}  // namespace RAJA

#endif
