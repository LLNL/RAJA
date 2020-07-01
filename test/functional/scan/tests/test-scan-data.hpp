//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_SCAN_DATA_HPP__
#define __TEST_SCAN_DATA_HPP__

//
// Methods to allocate/deallocate scan test data.
//

template <typename T>
void allocScanTestData(int N,
                       camp::resources::Resource& work_res,
                       T** work_in, T** work_out,
                       T** host_in, T** host_out)
{
  camp::resources::Resource host_res{camp::resources::Host()};

  *work_in  = work_res.allocate<T>(N);
  *work_out = work_res.allocate<T>(N);

  *host_in  = host_res.allocate<T>(N);
  *host_out = host_res.allocate<T>(N);
}

template <typename T>
void deallocScanTestData(camp::resources::Resource& work_res,
                         T* work_in, T* work_out,
                         T* host_in, T* host_out)
{
  camp::resources::Resource host_res{camp::resources::Host()};

  work_res.deallocate(work_in);
  work_res.deallocate(work_out);
  host_res.deallocate(host_in);
  host_res.deallocate(host_out);
}

#endif // __TEST_SCAN_DATA_HPP__
