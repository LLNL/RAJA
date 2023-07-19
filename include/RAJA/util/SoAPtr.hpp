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
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
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
template <typename T>
class SoAPtr
{
  using value_type = T;

public:
  SoAPtr() = default;
  template < typename Res >
  SoAPtr(size_t size, Res& res)
      : mem(res.template allocate<value_type>(size, ::RAJA::resources::MemoryAccess::Device))
  {
  }

  template < typename Res >
  SoAPtr& allocate(size_t size, Res& res)
  {
    mem = res.template allocate<value_type>(size, ::RAJA::resources::MemoryAccess::Device);
    return *this;
  }

  template < typename Res >
  SoAPtr& deallocate(Res& res)
  {
    res.deallocate(mem, ::RAJA::resources::MemoryAccess::Device);
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
template <typename T, typename IndexType, bool doing_min>
class SoAPtr<RAJA::reduce::detail::ValueLoc<T, IndexType, doing_min>>
{
  using value_type = RAJA::reduce::detail::ValueLoc<T, IndexType, doing_min>;
  using first_type = T;
  using second_type = IndexType;

public:
  SoAPtr() = default;
  template < typename Res >
  explicit SoAPtr(size_t size, Res& res)
      : mem(res.template allocate<first_type>(size, ::RAJA::resources::MemoryAccess::Device)),
        mem_idx(res.template allocate<second_type>(size, ::RAJA::resources::MemoryAccess::Device))
  {
  }

  template < typename Res >
  SoAPtr& allocate(size_t size, Res& res)
  {
    mem = res.template allocate<first_type>(size, ::RAJA::resources::MemoryAccess::Device);
    mem_idx = res.template allocate<second_type>(size, ::RAJA::resources::MemoryAccess::Device);
    return *this;
  }

  template < typename Res >
  SoAPtr& deallocate(Res& res)
  {
    res.deallocate(mem, ::RAJA::resources::MemoryAccess::Device);
    mem = nullptr;
    res.deallocate(mem_idx, ::RAJA::resources::MemoryAccess::Device);
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
