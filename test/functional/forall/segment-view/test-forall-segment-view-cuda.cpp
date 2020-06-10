//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "test-forall-segment-view.hpp"

#if defined(RAJA_ENABLE_CUDA)

// Cartesian product of types for Cuda tests
using CudaForallSegmentTypes = 
  Test< camp::cartesian_product<IdxTypeList, 
                                CudaResourceList, 
                                CudaForallExecPols> >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Cuda,
                               ForallSegmentViewTest,
                               CudaForallSegmentTypes);

#endif
