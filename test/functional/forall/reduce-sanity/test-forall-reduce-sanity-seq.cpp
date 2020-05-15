//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall-reduce-sanity.hpp"

#include "../test-reducepol.hpp"

// Sequential execution policy types
using SequentialForallReduceExecPols = camp::list< RAJA::seq_exec,
                                                   RAJA::loop_exec >;

// Cartesian product of types for Sequential tests
using SequentialForallReduceSanityTypes =
  Test< camp::cartesian_product<ReduceSanityDataTypeList, 
                                HostResourceList, 
                                SequentialForallReduceExecPols,
                                SequentialReducePols>>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Sequential,
                               ForallReduceSanityTest,
                               SequentialForallReduceSanityTypes);
