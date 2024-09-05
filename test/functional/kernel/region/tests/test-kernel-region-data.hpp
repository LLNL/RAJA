//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_REGION_DATA_HPP__
#define __TEST_KERNEL_REGION_DATA_HPP__

template <typename T>
void allocRegionTestData(int                       N,
                         camp::resources::Resource work_res,
                         T**                       work1,
                         T**                       work2,
                         T**                       work3,
                         camp::resources::Resource host_res,
                         T**                       check)
{
  *work1 = work_res.allocate<T>(N);
  *work2 = work_res.allocate<T>(N);
  *work3 = work_res.allocate<T>(N);

  *check = host_res.allocate<T>(N);
}

template <typename T>
void deallocRegionTestData(camp::resources::Resource work_res,
                           T*                        work1,
                           T*                        work2,
                           T*                        work3,
                           camp::resources::Resource host_res,
                           T*                        check)
{
  work_res.deallocate(work1);
  work_res.deallocate(work2);
  work_res.deallocate(work3);

  host_res.deallocate(check);
}

#endif // __TEST_KERNEL_REGION_UTILS_HPP__
