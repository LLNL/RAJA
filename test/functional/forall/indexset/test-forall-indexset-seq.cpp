//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall-indexset.hpp"

// Sequential execution policy types
using SequentialForallIndexSetExecPols = 
  camp::list< RAJA::ExecPolicy<RAJA::seq_segit, RAJA::seq_exec>,
              RAJA::ExecPolicy<RAJA::seq_segit, RAJA::loop_exec>,
              RAJA::ExecPolicy<RAJA::seq_segit, RAJA::simd_exec> >;

// Cartesian product of types for Sequential tests
using SequentialForallIndexSetTypes =
  Test< camp::cartesian_product<IdxTypeList, 
                                HostResourceList, 
                                SequentialForallIndexSetExecPols>>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Sequential,
                               ForallIndexSetTest,
                               SequentialForallIndexSetTypes);
