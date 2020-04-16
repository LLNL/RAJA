//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall-indexset.hpp"

// Hip execution policy types
using HipForallIndexSetExecPols = 
  camp::list< RAJA::ExecPolicy<RAJA::seq_segit, RAJA::hip_exec<128>>,
              RAJA::ExecPolicy<RAJA::seq_segit, RAJA::hip_exec<256>> >;

// Cartesian product of types for Hip tests
using HipForallIndexSetTypes =
  Test< camp::cartesian_product<IdxTypeList, 
                                HipResourceList, 
                                HipForallIndexSetExecPols>>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Hip,
                               ForallIndexSetTest,
                               HipForallIndexSetTypes);
