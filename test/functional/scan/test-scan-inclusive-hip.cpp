//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "test-scan-exclusive.hpp"

#if defined(RAJA_ENABLE_HIP)

// CUDA policy types to test
using HipExecTypes = list< RAJA::hip_exec<128>,
                           RAJA::hip_exec<256> >;

using ListHipRes = list<camp::resources::Hip>;

using HipInclusiveScanTypes = 
  Test<cartesian_product< HipExecTypes, ListHipRes, OpTypes >>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Hip, 
                               ScanFunctionalTest, 
                               HipInclusiveScanTypes);

#endif
