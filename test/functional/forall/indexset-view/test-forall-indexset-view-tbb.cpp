//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall-indexset-view.hpp"

// TBB execution policy types
using TBBForallIndexSetExecPols = 
  camp::list< RAJA::ExecPolicy<RAJA::tbb_for_exec, RAJA::seq_exec>,
              RAJA::ExecPolicy<RAJA::tbb_for_exec, RAJA::loop_exec>,
              RAJA::ExecPolicy<RAJA::tbb_for_exec, RAJA::simd_exec>,
              RAJA::ExecPolicy<RAJA::tbb_for_dynamic, RAJA::seq_exec>,
              RAJA::ExecPolicy<RAJA::tbb_for_dynamic, RAJA::loop_exec>,
              RAJA::ExecPolicy<RAJA::tbb_for_dynamic, RAJA::simd_exec>,
              RAJA::ExecPolicy<RAJA::seq_segit, RAJA::tbb_for_exec>,
              RAJA::ExecPolicy<RAJA::seq_segit, RAJA::tbb_for_static< >>,
              RAJA::ExecPolicy<RAJA::seq_segit, RAJA::tbb_for_static< 2 >>,
              RAJA::ExecPolicy<RAJA::seq_segit, RAJA::tbb_for_static< 4 >>,
              RAJA::ExecPolicy<RAJA::seq_segit, RAJA::tbb_for_static< 8 >>,
              RAJA::ExecPolicy<RAJA::seq_segit, RAJA::tbb_for_dynamic> >;

// Cartesian product of types for TBB tests
using TBBForallIndexSetTypes =
  Test< camp::cartesian_product<IdxTypeList, 
                                HostResourceList, 
                                TBBForallIndexSetExecPols>>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(TBB,
                               ForallIndexSetViewTest,
                               TBBForallIndexSetTypes);
