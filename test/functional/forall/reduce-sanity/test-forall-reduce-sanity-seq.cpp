//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "test-forall-reduce-sanity.hpp"

// Cartesian product of types for Sequential tests
using SequentialForallReduceSanityTypes =
  Test< camp::cartesian_product<ReduceSanityDataTypeList, 
                                HostResourceList, 
                                SequentialForallReduceExecPols,
                                SequentialReducePols>>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Sequential,
                               ForallReduceSanityTest,
                               SequentialForallReduceSanityTypes);
