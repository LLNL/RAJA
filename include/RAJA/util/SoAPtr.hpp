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
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_SOA_PTR_HPP
#define RAJA_SOA_PTR_HPP

#include "RAJA/config.hpp"

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
      : mem(mempool::getInstance().template malloc<value_type>(size))
  {
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
      : mem(mempool::getInstance().template malloc<first_type>(size)),
        mem_idx(mempool::getInstance().template malloc<second_type>(size))
  {
  }

  SoAPtr& allocate(size_t size)
  {
    mem = mempool::getInstance().template malloc<first_type>(size);
    mem_idx = mempool::getInstance().template malloc<second_type>(size);
    return *this;
  }

  SoAPtr& deallocate()
  {
    mempool::getInstance().free(mem);
    mem = nullptr;
    mempool::getInstance().free(mem_idx);
    mem_idx = nullptr;
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
