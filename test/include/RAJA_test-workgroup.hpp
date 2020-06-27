//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_WORKGROUP_UTILS_HPP__
#define __TEST_WORKGROUP_UTILS_HPP__

#include "RAJA_test-base.hpp"
#include "RAJA_test-camp.hpp"

#include "RAJA_unit-test-forone.hpp"

#include <cstddef>
#include <limits>
#include <new>

namespace detail {

template < typename Resource >
struct ResourceAllocator
{
  template < typename T >
  struct std_allocator
  {
    using value_type = T;

    std_allocator() = default;

    template < typename U >
    std_allocator(std_allocator<U> const& other) noexcept
      : m_res(other.get_resource())
    { }

    /*[[nodiscard]]*/
    value_type* allocate(size_t num)
    {
      if (num > std::numeric_limits<size_t>::max() / sizeof(value_type)) {
        throw std::bad_alloc();
      }

      value_type* ptr = m_res.template allocate<value_type>(num);

      if (!ptr) {
        throw std::bad_alloc();
      }

      return ptr;
    }

    void deallocate(value_type* ptr, size_t) noexcept
    {
      m_res.deallocate(ptr);
    }

    Resource const& get_resource() const
    {
      return m_res;
    }

  private:
    Resource m_res;
  };
};

template <typename T, typename U, typename Resource>
bool operator==(
    typename ResourceAllocator<Resource>::template std_allocator<T> const& lhs,
    typename ResourceAllocator<Resource>::template std_allocator<U> const& rhs)
{
  return lhs.get_resource() == rhs.get_resource();
}

template <typename T, typename U, typename Resource>
bool operator!=(
    typename ResourceAllocator<Resource>::template std_allocator<T> const& lhs,
    typename ResourceAllocator<Resource>::template std_allocator<U> const& rhs)
{
  return !(lhs == rhs);
}

} // namespace detail

//
// Data types
//
using IndexTypeTypeList = camp::list<
                                 int,
                                 long,
                                 RAJA::Index_type
                               >;

using XargsTypeList = camp::list<
                                 RAJA::xargs<>,
                                 RAJA::xargs<int*>,
                                 RAJA::xargs<int, int*>
                               >;

using SequentialExecPolicyList =
    camp::list<
                RAJA::loop_work,
                RAJA::seq_work
              >;
using SequentialOrderedPolicyList =
    camp::list<
                RAJA::ordered,
                RAJA::reverse_ordered
              >;
using SequentialOrderPolicyList =
    camp::list<
                RAJA::ordered,
                RAJA::reverse_ordered
              >;
using SequentialStoragePolicyList =
    camp::list<
                RAJA::array_of_pointers,
                RAJA::ragged_array_of_objects,
                RAJA::constant_stride_array_of_objects
              >;

#if defined(RAJA_ENABLE_TBB)
using TBBExecPolicyList =
    camp::list<
                RAJA::tbb_work
              >;
using TBBOrderedPolicyList = SequentialOrderedPolicyList;
using TBBOrderPolicyList   = SequentialOrderPolicyList;
using TBBStoragePolicyList = SequentialStoragePolicyList;
#endif

#if defined(RAJA_ENABLE_OPENMP)
using OpenMPExecPolicyList =
    camp::list<
                RAJA::omp_work
              >;
using OpenMPOrderedPolicyList = SequentialOrderedPolicyList;
using OpenMPOrderPolicyList   = SequentialOrderPolicyList;
using OpenMPStoragePolicyList = SequentialStoragePolicyList;
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OpenMPTargetExecPolicyList =
    camp::list<
                RAJA::omp_target_work
              >;
using OpenMPTargetOrderedPolicyList = SequentialOrderedPolicyList;
using OpenMPTargetOrderPolicyList   = SequentialOrderPolicyList;
using OpenMPTargetStoragePolicyList = SequentialStoragePolicyList;
#endif

#if defined(RAJA_ENABLE_CUDA)
using CudaExecPolicyList =
    camp::list<
                RAJA::cuda_work<256>,
                RAJA::cuda_work<1024>
              >;
using CudaOrderedPolicyList = SequentialOrderedPolicyList;
using CudaOrderPolicyList   =
    camp::list<
                RAJA::ordered,
                RAJA::reverse_ordered,
                RAJA::unordered_cuda_loop_y_block_iter_x_threadblock_average
              >;
using CudaStoragePolicyList = SequentialStoragePolicyList;
#endif

#if defined(RAJA_ENABLE_HIP)
using HipExecPolicyList =
    camp::list<
                RAJA::hip_work<256>,
                RAJA::hip_work<1024>
              >;
using HipOrderedPolicyList = SequentialOrderedPolicyList;
using HipOrderPolicyList   = SequentialOrderPolicyList;
using HipStoragePolicyList = SequentialStoragePolicyList;
#endif


//
// Memory resource types for beck-end execution
//
using HostAllocatorList = camp::list<typename detail::ResourceAllocator<camp::resources::Host>::template std_allocator<char>>;

using SequentialAllocatorList = HostAllocatorList;

#if defined(RAJA_ENABLE_TBB)
using TBBAllocatorList = HostAllocatorList;
#endif

#if defined(RAJA_ENABLE_OPENMP)
using OpenMPAllocatorList = HostAllocatorList;
#endif

#if defined(RAJA_ENABLE_CUDA)
using CudaAllocatorList = camp::list<typename detail::ResourceAllocator<camp::resources::Cuda>::template std_allocator<char>>;
#endif

#if defined(RAJA_ENABLE_HIP)
using HipAllocatorList = camp::list<typename detail::ResourceAllocator<camp::resources::Hip>::template std_allocator<char>>;
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OpenMPTargetAllocatorList = camp::list<typename detail::ResourceAllocator<camp::resources::Omp>::template std_allocator<char>>;
#endif

#endif  // __TEST_WORKGROUP_UTILS_HPP__
