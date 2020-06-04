//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall-reduce-sanity-tests.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "../test-forall-execpol.hpp"
#include "../test-reducepol.hpp"
#include "../test-forall-utils.hpp"

// Cartesian product of types for HIP tests
using HipForallReduceSanityTypes =
  Test< camp::cartesian_product<ReductionDataTypeList, 
                                HipResourceList, 
                                HipForallExecPols,
                                HipReducePols>>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Hip,
                               ForallReduceSumSanityTest,
                               HipForallReduceSanityTypes);

INSTANTIATE_TYPED_TEST_SUITE_P(Hip,
                               ForallReduceMinSanityTest,
                               HipForallReduceSanityTypes);

INSTANTIATE_TYPED_TEST_SUITE_P(Hip,
                               ForallReduceMaxSanityTest,
                               HipForallReduceSanityTypes);

INSTANTIATE_TYPED_TEST_SUITE_P(Hip,
                               ForallReduceMinLocSanityTest,
                               HipForallReduceSanityTypes);

INSTANTIATE_TYPED_TEST_SUITE_P(Hip,
                               ForallReduceMaxLocSanityTest,
                               HipForallReduceSanityTypes);

#endif
