//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall-reduce-sanity-tests.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "../test-forall-execpol.hpp"
#include "../test-reducepol.hpp"
#include "../test-forall-utils.hpp"

// Cartesian product of types for CUDA tests
using CudaForallReduceSanityTypes =
  Test< camp::cartesian_product<ReductionDataTypeList, 
                                CudaResourceList, 
                                CudaForallExecPols,
                                CudaReducePols>>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Cuda,
                               ForallReduceSumSanityTest,
                               CudaForallReduceSanityTypes);

INSTANTIATE_TYPED_TEST_SUITE_P(Cuda,
                               ForallReduceMinSanityTest,
                               CudaForallReduceSanityTypes);

INSTANTIATE_TYPED_TEST_SUITE_P(Cuda,
                               ForallReduceMaxSanityTest,
                               CudaForallReduceSanityTypes);

INSTANTIATE_TYPED_TEST_SUITE_P(Cuda,
                               ForallReduceMinLocSanityTest,
                               CudaForallReduceSanityTypes);

INSTANTIATE_TYPED_TEST_SUITE_P(Cuda,
                               ForallReduceMaxLocSanityTest,
                               CudaForallReduceSanityTypes);

#endif
