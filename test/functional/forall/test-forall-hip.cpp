//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall-segment.hpp"

#if defined(RAJA_ENABLE_HIP)

// Hip execution policy types
using HipTypes = list< RAJA::hip_exec<128>,
                       RAJA::hip_exec<256>  >;

// Hip tests index, resource, and execution policy types
using ListHip = list<camp::resources::Hip>;
using HipForallTypes =
    Test<cartesian_product<IdxTypes, ListHip, HipTypes>>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Hip,
                               ForallSegmentTest,
                               HipForallTypes);

#endif
