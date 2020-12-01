//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Data management used for loc reductions.
//

#ifndef __RAJA_test_reduceloc_data_HPP__
#define __RAJA_test_reduceloc_data_HPP__

#include "camp/list.hpp"

template<typename T>
void allocateReduceLocTestData(int N,
                            camp::resources::Resource& work_res,
                            T** work_array,
                            T** check_array)
{
  camp::resources::Resource host_res{camp::resources::Host()};

  *work_array = work_res.allocate<T>(N);

  *check_array = host_res.allocate<T>(N);
}

template<typename T>
void deallocateReduceLocTestData(camp::resources::Resource& work_res,
                              T* work_array,
                              T* check_array)
{
  camp::resources::Resource host_res{camp::resources::Host()};

  work_res.deallocate(work_array);

  host_res.deallocate(check_array);
}

#endif // __RAJA_test_reduceloc_data_HPP__
