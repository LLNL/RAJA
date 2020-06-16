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

#endif  // __TEST_WORKGROUP_UTILS_HPP__
