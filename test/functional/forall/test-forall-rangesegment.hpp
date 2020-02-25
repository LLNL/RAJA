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

using camp::resources::Host;

template<typename INDEX_TYPE, typename WORKING_RES, typename EXEC_POLICY>
void ForallRangeSegmentFunctionalTest(INDEX_TYPE first, INDEX_TYPE last)
{
  RAJA::TypedRangeSegment<INDEX_TYPE> r1(first, last);
  RAJA::Index_type N = r1.end() - r1.begin();

  camp::resources::Resource working_res{WORKING_RES()};
  camp::resources::Resource check_res{Host()};

  INDEX_TYPE * working_array = working_res.allocate<INDEX_TYPE>(N);
  INDEX_TYPE * check_array   = check_res.allocate<INDEX_TYPE>(N);
  INDEX_TYPE * test_array    = check_res.allocate<INDEX_TYPE>(N);

  for(INDEX_TYPE i=0; i < r1.size(); i++)
  {
    test_array[i]= *r1.begin() + i;
  }

  RAJA::forall<EXEC_POLICY>(r1,
    [=] RAJA_HOST_DEVICE (INDEX_TYPE idx){
    working_array[idx - *r1.begin()] = idx;
  });

  working_res.memcpy(check_array, working_array, sizeof(INDEX_TYPE) * N );

  for(INDEX_TYPE i=0; i < r1.size(); i++)
  {
    ASSERT_EQ(test_array[i], check_array[i]);
  }

  working_res.deallocate(working_array);
  check_res.deallocate(check_array);
  check_res.deallocate(test_array);
}
#endif // __TEST_FORALL_RANGESEGMENT_HPP__
