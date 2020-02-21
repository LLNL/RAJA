//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_RANGESEGMENT_HPP__
#define __TEST_FORALL_RANGESEGMENT_HPP__

#include "RAJA/RAJA.hpp"
#include "camp/resource.hpp"

template<typename T>
void ForallRangeSegmentFunctionalTest_host(T first, T last)
{
  RAJA::TypedRangeSegment<T> r1(first, last);
  RAJA::Index_type N = r1.end() - r1.begin();

  camp::resources::Resource res{camp::resources::Host()};
  T * ref_array = res.allocate<T>(N);

  for(RAJA::Index_type i=0; i < r1.size(); i++)
  {
    ref_array[i]= *r1.begin() + i;
  }

  RAJA::forall<RAJA::seq_exec>(r1, [=](RAJA::Index_type idx){
    ASSERT_EQ(ref_array[idx-*r1.begin()], idx);
  });

  res.deallocate(ref_array);
}

#if defined(RAJA_ENABLE_CUDA)
template<typename T>
void ForallRangeSegmentFunctionalTest_cuda(T first, T last)
{
  RAJA::TypedRangeSegment<T> r1(first, last);
  RAJA::Index_type N = r1.end() - r1.begin();

  camp::resources::Resource dev_res{camp::resources::Cuda()};
  T * d_ref_array = dev_res.allocate<T>(N);

  camp::resources::Resource host_res{camp::resources::Host()};
  T * ref_array = host_res.allocate<T>(N);
  T * test_array = host_res.allocate<T>(N);

  for(RAJA::Index_type i=0; i < r1.size(); i++)
  {
    test_array[i]= *r1.begin() + i;
  }

  RAJA::forall<RAJA::cuda_exec<256>>(r1,
    [=] RAJA_DEVICE (int idx){
    d_ref_array[idx - *r1.begin()] = idx;
  });

  dev_res.memcpy(ref_array, d_ref_array, N*sizeof(T));

  for(RAJA::Index_type i=0; i < r1.size(); i++)
  {
    ASSERT_EQ(test_array[i], ref_array[i]);
  }

  host_res.deallocate(ref_array);
  host_res.deallocate(test_array);
  dev_res.deallocate(d_ref_array);
  
}
#endif

#endif // __TEST_FORALL_RANGESEGMENT_HPP__
