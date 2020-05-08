//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-kernel-region.hpp"

#include "../../forall/test-forall-utils.hpp"

using SequentialKernelRegionExecPols = 
  camp::list< 

    RAJA::KernelPolicy<
      RAJA::statement::Region<RAJA::seq_region,
        RAJA::statement::For<0, RAJA::seq_exec,
          RAJA::statement::Lambda<0>
        >,
        RAJA::statement::For<0, RAJA::seq_exec,
          RAJA::statement::Lambda<1>
        >,
        RAJA::statement::For<0, RAJA::seq_exec,
          RAJA::statement::Lambda<2>
        >
      >
    >,

    RAJA::KernelPolicy<
      RAJA::statement::Region<RAJA::seq_region,
        RAJA::statement::For<0, RAJA::loop_exec,
          RAJA::statement::Lambda<0>
        >,
        RAJA::statement::For<0, RAJA::loop_exec,
          RAJA::statement::Lambda<1>
        >,
        RAJA::statement::For<0, RAJA::loop_exec,
          RAJA::statement::Lambda<2>
        >
      >
    >

  >;


// Cartesian product of types for Sequential tests
using SequentialKernelRegionTypes =
  Test< camp::cartesian_product<IdxTypeList, 
                                HostResourceList,
                                SequentialKernelRegionExecPols> >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Sequential,
                               KernelRegionFunctionalTest,
                               SequentialKernelRegionTypes);
