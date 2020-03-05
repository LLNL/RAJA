//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"

#include "test-forall-rangesegment.hpp"

// Generate Cuda Type List
#if defined(RAJA_ENABLE_CUDA)
using CudaTypes = list< RAJA::cuda_exec<128>
                      >;

using ListCuda = list < camp::resources::Cuda >;
using CudaForallTypes = Test<cartesian_product< IdxTypes, ListCuda, CudaTypes >>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Cuda, ForallFunctionalTest, CudaForallTypes);
#endif
