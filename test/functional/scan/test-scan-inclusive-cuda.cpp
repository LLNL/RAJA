//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-scan-inclusive.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "../forall/test-forall-utils.hpp"
#include "../forall/test-forall-execpol.hpp"

using CudaInclusiveScanTypes =
  Test<camp::cartesian_product< CudaForallExecPols, 
                                CudaResourceList, 
                                ScanOpTypes >>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Cuda, 
                               ScanFunctionalTest, 
                               CudaInclusiveScanTypes);

#endif
