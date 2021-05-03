/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for common RAJA internal definitions.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_SOA_PTR_HPP
#define RAJA_SOA_PTR_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/align.hpp"
#include "RAJA/util/macros.hpp"

// for RAJA::reduce::detail::ValueLoc
#include "RAJA/pattern/detail/reduce.hpp"

namespace RAJA
{

namespace detail
{

/*!
 * @brief Pointer class specialized for Struct of Array data layout allocated
 *        via RAJA basic_mempools.
 *
 * This is useful for creating a vectorizable data layout and getting
 * coalesced memory accesses or avoiding shared memory bank conflicts in cuda.
 */
template <typename T,
          typename mempool = RAJA::basic_mempool::MemPool<
              RAJA::basic_mempool::generic_allocator> >
class SoAPtr
{
  using value_type = T;

public:
  SoAPtr() = default;
  explicit SoAPtr(size_t size)
  {
    allocate(size);
  }

  size_t allocationSize(size_t size) const
  {
    return sizeof(value_type) * size;
  }

  SoAPtr& setMemory(size_t size, void* memory)
  {
    mem = static_cast<value_type*>(memory);
    return *this;
  }

  SoAPtr& forgetMemory()
  {
    mem = nullptr;
    return *this;
  }

  SoAPtr& allocate(size_t size)
  {
    mem = mempool::getInstance().template malloc<value_type>(size);
    return *this;
  }

  SoAPtr& deallocate()
  {
    mempool::getInstance().free(mem);
    mem = nullptr;
    return *this;
  }

  RAJA_HOST_DEVICE bool allocated() const { return mem != nullptr; }

  RAJA_HOST_DEVICE value_type get(size_t i) const { return mem[i]; }

  RAJA_HOST_DEVICE void set(size_t i, value_type val) { mem[i] = val; }

private:
  value_type* mem = nullptr;
};

/*!
 * @brief Specialization for RAJA::reduce::detail::ValueLoc.
 */
template <typename T, typename IndexType, bool doing_min, typename mempool>
class SoAPtr<RAJA::reduce::detail::ValueLoc<T, IndexType, doing_min>, mempool>
{
  using value_type = RAJA::reduce::detail::ValueLoc<T, IndexType, doing_min>;
  using first_type = T;
  using second_type = IndexType;

public:
  SoAPtr() = default;
  explicit SoAPtr(size_t size)
  {
    allocate(size);
  }

  size_t allocationSize(size_t size) const
  {
    return (sizeof(first_type) + sizeof(second_type)) * size
         + alignof(second_type);
  }

  SoAPtr& setMemory(size_t size, void* memory)
  {
    const size_t first_size = sizeof(first_type) * size;
    const size_t second_size = sizeof(second_type) * size;
    size_t second_capacity = allocationSize(size) - first_size;

    mem     = static_cast<first_type*>(memory);
    void* second_memory = static_cast<void*>(static_cast<char*>(memory) + first_size);
    mem_idx = static_cast<second_type*>(RAJA::align(
        alignof(second_type), second_size, second_memory, second_capacity));
    if (mem_idx == nullptr) {
      RAJA_ABORT_OR_THROW("SoAPtr unable to align second memory");
    }
    return *this;
  }

  SoAPtr& forgetMemory()
  {
    mem_idx = nullptr;
    mem     = nullptr;
    return *this;
  }

  SoAPtr& allocate(size_t size)
  {
    mem     = mempool::getInstance().template malloc<first_type>(size);
    mem_idx = mempool::getInstance().template malloc<second_type>(size);
    return *this;
  }

  SoAPtr& deallocate()
  {
    mempool::getInstance().free(mem_idx);
    mem_idx = nullptr;
    mempool::getInstance().free(mem);
    mem = nullptr;
    return *this;
  }

  RAJA_HOST_DEVICE bool allocated() const { return mem != nullptr; }

  RAJA_HOST_DEVICE value_type get(size_t i) const
  {
    return value_type(mem[i], mem_idx[i]);
  }

  RAJA_HOST_DEVICE void set(size_t i, value_type val)
  {
    mem[i] = val;
    mem_idx[i] = val.getLoc();
  }

private:
  first_type* mem = nullptr;
  second_type* mem_idx = nullptr;
};

}  // namespace detail

}  // namespace RAJA

#endif /* RAJA_SOA_PTR_HPP */
