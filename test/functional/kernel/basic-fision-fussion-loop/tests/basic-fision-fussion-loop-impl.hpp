
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __BASIC_FISION_FUSSION_LOOP_SEGMENTS_IMPL_HPP__
#define __BASIC_FISION_FUSSION_LOOP_SEGMENTS_IMPL_HPP__

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <numeric>
#include <vector>

template <typename IDX_TYPE, typename EXEC_POLICY, typename WORKING_RES, typename SEG_TYPE>
void KernelBasicFisionFussionLoopTestImpl(const SEG_TYPE& seg,              
                                         WORKING_RES working_res,
                                         camp::resources::Resource erased_working_res)
{
  IDX_TYPE data_len = IDX_TYPE(0);

  std::vector<IDX_TYPE> seg_idx;
  RAJA::getIndices(seg_idx, seg);
  if ( seg_idx.size() > 0 ) {
    data_len = seg_idx[seg_idx.size() - 1] + 1;
  }

  IDX_TYPE* working_array_x;
  IDX_TYPE* working_array_y;
  IDX_TYPE* check_array_x;
  IDX_TYPE* check_array_y;
  IDX_TYPE* test_array_x;
  IDX_TYPE* test_array_y;

  allocateForallTestData<IDX_TYPE>(data_len,
                                   erased_working_res,
                                   &working_array_x,
                                   &check_array_x,
                                   &test_array_x);

  allocateForallTestData<IDX_TYPE>(data_len,
                                   erased_working_res,
                                   &working_array_y,
                                   &check_array_y,
                                   &test_array_y);
 

  memset(static_cast<void*>(test_array_x), 0, 
         sizeof(IDX_TYPE) * RAJA::stripIndexType(data_len));


  RAJA::kernel<EXEC_POLICY>(RAJA::make_tuple(seg, seg),
                            
                            [=] RAJA_HOST_DEVICE(IDX_TYPE i) { 
                              working_array_x[RAJA::stripIndexType(i)] += 1;
                            },
                            
                            [=] RAJA_HOST_DEVICE(IDX_TYPE i) { 
                              working_array_x[RAJA::stripIndexType(i)] += 2; }                            
                            );

  working_res.memcpy(check_array_x, working_array_x, 
                     sizeof(IDX_TYPE) * RAJA::stripIndexType(data_len));

  memset(static_cast<void*>(check_array_y), 0, 
         sizeof(IDX_TYPE) * RAJA::stripIndexType(data_len));

  RAJA::forall<RAJA::loop_exec>(working_res, seg, [=] (IDX_TYPE i) {
      check_array_y[RAJA::stripIndexType(i)] += 1;
      check_array_y[RAJA::stripIndexType(i)] += 2;
  });


  for (IDX_TYPE i = IDX_TYPE(0); i < data_len; ++i) {
    ASSERT_EQ( check_array_x[RAJA::stripIndexType(i)],
               check_array_y[RAJA::stripIndexType(i)] );
  }

  deallocateForallTestData<IDX_TYPE>(erased_working_res,
                                     working_array_x,
                                     check_array_x,
                                     test_array_x);


  deallocateForallTestData<IDX_TYPE>(erased_working_res,
                                     working_array_y,
                                     check_array_y,
                                     test_array_y);
}

#endif  // __BASIC_FISION_FUSSION_LOOP_SEGMENTS_IMPL_HPP__
