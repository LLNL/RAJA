//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_WORKGROUP_UTILS_HPP__
#define __TEST_WORKGROUP_UTILS_HPP__

#include "camp/resource.hpp"
#include "gtest/gtest.h"

#include "RAJA_unit_forone.hpp"

//
// Unroll types for gtest testing::Types
//
template <class T>
struct Test;

template <class... T>
struct Test<camp::list<T...>> {
  using Types = ::testing::Types<T...>;
};

namespace detail {

template < typename Resource >
struct ResourceAllocator
{
  void* allocate(size_t size)
  {
    return res.template allocate<char>(size);
  }
  void deallocate(void* ptr)
  {
    res.deallocate(ptr);
  }
private:
  Resource res;
};

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
                RAJA::seq_work
              >;
using SequentialOrderPolicyList =
    camp::list<
                RAJA::ordered,
                RAJA::unordered
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
using TBBOrderPolicyList = SequentialOrderPolicyList;
using TBBStoragePolicyList = SequentialStoragePolicyList;
#endif

#if defined(RAJA_ENABLE_OPENMP)
using OpenMPExecPolicyList =
    camp::list<
                RAJA::omp_work
              >;
using OpenMPOrderPolicyList = SequentialOrderPolicyList;
using OpenMPStoragePolicyList = SequentialStoragePolicyList;
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OpenMPTargetExecPolicyList =
    camp::list<
                RAJA::omp_target_work
              >;
using OpenMPTargetOrderPolicyList = SequentialOrderPolicyList;
using OpenMPTargetStoragePolicyList = SequentialStoragePolicyList;
#endif

#if defined(RAJA_ENABLE_CUDA)
using CudaExecPolicyList =
    camp::list<
                RAJA::cuda_work
              >;
using CudaOrderPolicyList = SequentialOrderPolicyList;
using CudaStoragePolicyList = SequentialStoragePolicyList;
#endif

#if defined(RAJA_ENABLE_HIP)
using HipExecPolicyList =
    camp::list<
                RAJA::hip_work
              >;
using HipOrderPolicyList = SequentialOrderPolicyList;
using HipStoragePolicyList = SequentialStoragePolicyList;
#endif


//
// Memory resource types for beck-end execution
//
using HostAllocatorList = camp::list<detail::ResourceAllocator<camp::resources::Host>>;

#if defined(RAJA_ENABLE_CUDA)
using CudaAllocatorList = camp::list<detail::ResourceAllocator<camp::resources::Cuda>>;
#endif

#if defined(RAJA_ENABLE_HIP)
using HipAllocatorList = camp::list<detail::ResourceAllocator<camp::resources::Hip>>;
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OpenMPTargetAllocatorList = camp::list<detail::ResourceAllocator<camp::resources::Omp>>;
#endif


//
// Memory resource types for beck-end execution
//
using HostResourceList = camp::list<camp::resources::Host>;

#if defined(RAJA_ENABLE_CUDA)
using CudaResourceList = camp::list<camp::resources::Cuda>;
#endif

#if defined(RAJA_ENABLE_HIP)
using HipResourceList = camp::list<camp::resources::Hip>;
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OpenMPTargetResourceList = camp::list<camp::resources::Omp>;
#endif


//
// Forone unit test policies
//
using SequentialForoneList = camp::list<forone_seq>;

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OpenmpTargetForoneList = camp::list<forone_openmp_target>;
#endif

#if defined(RAJA_ENABLE_CUDA)
using CudaForoneList = camp::list<forone_cuda>;
#endif

#if defined(RAJA_ENABLE_HIP)
using HipForoneList = camp::list<forone_hip>;
#endif

#endif  // __TEST_WORKGROUP_UTILS_HPP__
