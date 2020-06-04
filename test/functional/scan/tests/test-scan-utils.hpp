//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_SCAN_UTILS_HPP__
#define __TEST_SCAN_UTILS_HPP__

#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"

#include "camp/list.hpp"


// Scan functional test class
template<typename T>
class ScanFunctionalTest: public ::testing::Test {};


// Define scan operation types
using ScanOpTypes = camp::list< RAJA::operators::plus<int>,
#if 0  // Parallel tests with plus operator and float data do not work
       // likely due to precision being too low and plus not associative
                                RAJA::operators::plus<float>,
#endif
                                RAJA::operators::plus<double>,
                                RAJA::operators::minimum<int>,
                                RAJA::operators::minimum<float>,
                                RAJA::operators::minimum<double>,
                                RAJA::operators::maximum<int>,
                                RAJA::operators::maximum<float>,
                                RAJA::operators::maximum<double> >;

using ListHostRes = camp::list< camp::resources::Host >;


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



TYPED_TEST_SUITE_P(ScanFunctionalTest);

#endif //__TEST_SCAN_UTILS_HPP__
