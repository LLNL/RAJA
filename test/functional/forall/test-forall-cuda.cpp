//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall.hpp"

#if defined(RAJA_ENABLE_CUDA)

// Cuda execution policy types
using CudaTypes = list< RAJA::cuda_exec<128>,
                        RAJA::cuda_exec<256> >;

// Cuda tests index, resource, and execution policy types 
using ListCuda = list<camp::resources::Cuda>;
using CudaForallTypes =
    Test<cartesian_product<IdxTypes, ListCuda, CudaTypes>>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Cuda,
                               ForallFunctionalSegmentTest,
                               CudaForallTypes);

#endif
