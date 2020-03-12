//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "test-forall-rangesegment.hpp"

// Generate Hip Type List
#if defined(RAJA_ENABLE_HIP)
using HipTypes = list< RAJA::hip_exec<128>
                      >;

using ListHip = list < camp::resources::Hip >;
using HipForallTypes = Test<cartesian_product< IdxTypes, ListHip, HipTypes >>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Hip, ForallFunctionalTest, HipForallTypes);
#endif
