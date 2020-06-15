//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall-indexset.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "../test-forall-indexset-execpol.hpp"

// Cartesian product of types for Sycl tests
using SyclForallIndexSetTypes =
  Test< camp::cartesian_product<IdxTypeList, 
                                SyclResourceList, 
                                SyclForallIndexSetExecPols>>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Sycl,
                               ForallIndexSetTest,
                               SyclForallIndexSetTypes);

#endif
