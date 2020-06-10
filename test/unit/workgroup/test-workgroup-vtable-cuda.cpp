//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA workgroup vtable.
///

#include "tests/test-workgroup-vtable.hpp"

#include "test-workgroup-utils.hpp"

#if defined(RAJA_ENABLE_CUDA)
using CudaBasicWorkGroupVtableTypes =
  Test< camp::cartesian_product< CudaExecPolicyList,
                                 IndexTypeTypeList,
                                 XargsTypeList,
                                 CudaResourceList,
                                 CudaForoneList > >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(CudaBasicTest,
                               WorkGroupBasicVtableUnitTest,
                               CudaBasicWorkGroupVtableTypes);
#endif

