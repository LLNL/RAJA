//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall-segment.hpp"

#if defined(RAJA_ENABLE_CUDA)

// Cuda execution policy types
using CudaForallExecPols = list< RAJA::cuda_exec<128>,
                                 RAJA::cuda_exec<256> >;

// Cartesian product of types for Cuda tests
using CudaForallSegmentTypes = 
  Test< cartesian_product<IdxTypeList, 
                          CudaResourceList, 
                          CudaForallExecPols> >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Cuda,
                               ForallSegmentTest,
                               CudaForallExecPols);

#endif
