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
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_UTIL_ALLOCATORUMPIRE_HPP
#define RAJA_UTIL_ALLOCATORUMPIRE_HPP

#include <string>

#include "RAJA/util/Allocator.hpp"

namespace RAJA
{

/*! \class AllocatorUmpire
 ******************************************************************************
 *
 * \brief  AllocatorUmpire wraps an umpire allocator so it can be used in RAJA.
 *         This is intended to be used with umpire memory pools.
 *
 * Example:
 *
 * umpire::Allocator umpire_allocator;
 *
 * RAJA::AllocatorUmpire<umpire::Allocator> raja_alocator(umpire_allocator);
 *
 ******************************************************************************
 */
template <typename allocator_t>
struct AllocatorUmpire : Allocator
{
  using allocator_type = allocator_t;

  AllocatorUmpire(allocator_type const& aloc = allocator_type{})
    : m_alloc(aloc)
    , m_name(std::string("RAJA::AllocatorUmpire<")+m_alloc.getName()+">")
  {
  }

  virtual ~AllocatorUmpire()
  {
    // With static objects like AllocatorUmpire, cudaErrorCudartUnloading is a possible
    // error with cudaFree
    // So no more cuda calls here
  }

  void* allocate(size_t nbytes) final
  {
    return m_alloc.allocate(nbytes);
  }

  void deallocate(void* ptr) final
  {
    return m_alloc.deallocate(ptr);
  }

  void release() final
  {
    m_alloc.release();
  }

  size_t getHighWatermark() const noexcept final
  {
    return m_alloc.getHighWatermark();
  }

  size_t getCurrentSize() const noexcept final
  {
    return m_alloc.getCurrentSize();
  }

  size_t getActualSize() const noexcept final
  {
    return m_alloc.getActualSize();
  }

  size_t getAllocationCount() const noexcept final
  {
    return m_alloc.getAllocationCount();
  }

  const std::string& getName() const noexcept final
  {
    return m_name;
  }

  Platform getPlatform() noexcept final
  {
    return m_alloc.getPlatform();
  }

private:
  allocator_t m_alloc;
  std::string m_name;
};

} /* end namespace RAJA */


#endif /* RAJA_UTIL_ALLOCATORUMPIRE_HPP */
