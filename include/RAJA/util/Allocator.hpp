/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing an implementation of a memory pool.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_UTIL_ALLOCATOR_HPP
#define RAJA_UTIL_ALLOCATOR_HPP

#include <cstddef>
#include <string>
#include <vector>

#include "RAJA/util/camp_aliases.hpp"

namespace RAJA
{

struct Allocator;

namespace detail
{

inline std::vector<Allocator*>& get_allocators();

} /* end namespace detail */

/*! \class Allocator
 ******************************************************************************
 *
 * \brief  Allocator Provides a generic interface for allocation in RAJA
 *
 * Allocator& device_pool = RAJA::cuda::get_device_allocator();
 *
 ******************************************************************************
 */
struct Allocator
{
  Allocator()
  {
    detail::get_allocators().emplace_back(this);
  }

  // not copyable or movable
  Allocator(Allocator const&) = delete;
  Allocator(Allocator &&) = delete;
  Allocator& operator=(Allocator const&) = delete;
  Allocator& operator=(Allocator &&) = delete;

  virtual ~Allocator()
  {
    auto& allocators = detail::get_allocators();
    for (auto iter = allocators.cbegin();
         iter != allocators.cend();
         ++iter) {
      if (this == *iter) {
        allocators.erase(iter);
        break;
      }
    }
  }

  virtual void* allocate(size_t nbytes,
                         size_t alignment = alignof(std::max_align_t)) = 0;

  template <typename T>
  inline T* allocate(size_t nitems,
                     size_t alignment = alignof(T))
  {
    return static_cast<T*>(this->allocate(sizeof(T)*nitems, alignment));
  }

  virtual void deallocate(void* ptr) = 0;

  virtual void release() = 0;

  virtual size_t getHighWatermark() const noexcept = 0;

  virtual size_t getCurrentSize() const noexcept = 0;

  virtual size_t getActualSize() const noexcept = 0;

  virtual size_t getAllocationCount() const noexcept = 0;

  virtual const std::string& getName() const noexcept = 0;

  // virtual Platform getPlatform() const noexcept = 0;
};

namespace detail
{

inline std::vector<Allocator*>& get_allocators()
{
  static std::vector<Allocator*> allocators;
  return allocators;
}

} /* end namespace detail */

/*!
 * \brief Get the set of allocators used by RAJA internally
 */
inline std::vector<Allocator*> const& get_allocators()
{
  return detail::get_allocators();
}

} /* end namespace RAJA */


#endif /* RAJA_UTIL_ALLOCATOR_HPP */
