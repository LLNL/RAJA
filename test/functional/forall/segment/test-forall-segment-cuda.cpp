//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall-segment.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "../test-forall-execpol.hpp"

// Cartesian product of types for Cuda tests
using CudaForallSegmentTypes = 
  Test< camp::cartesian_product<IdxTypeList, 
                                CudaResourceList, 
                                CudaForallExecPols> >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Cuda,
                               ForallSegmentTest,
                               CudaForallSegmentTypes);

#endif
