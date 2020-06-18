//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA workgroup unordered runs.
///

#include "tests/test-workgroup-unordered.hpp"

#if defined(RAJA_ENABLE_CUDA)
using CudaBasicWorkGroupUnorderedTypes =
  Test< camp::cartesian_product< CudaExecPolicyList,
                                 CudaOrderPolicyList,
                                 CudaStoragePolicyList,
                                 IndexTypeTypeList,
                                 CudaAllocatorList,
                                 CudaResourceList > >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(CudaBasicTest,
                               WorkGroupBasicUnorderedFunctionalTest,
                               CudaBasicWorkGroupUnorderedTypes);
#endif
