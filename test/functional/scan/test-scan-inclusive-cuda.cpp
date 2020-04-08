//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "test-scan-inclusive.hpp"

#if defined(RAJA_ENABLE_CUDA)

// CUDA policy types to test
using CudaExecTypes = camp::list< RAJA::cuda_exec<128>,
                                  RAJA::cuda_exec<256> >;

using ListCudaRes = camp::list<camp::resources::Cuda>;

using CudaInclusiveScanTypes =
  Test<cartesian_product< CudaExecTypes, ListCudaRes, OpTypes >>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Cuda, 
                               ScanFunctionalTest, 
                               CudaInclusiveScanTypes);

#endif
