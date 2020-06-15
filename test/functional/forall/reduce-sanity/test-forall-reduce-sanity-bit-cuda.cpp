//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall-reduce-sanity.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "../test-reducepol.hpp"
#include "../test-forall-execpol.hpp"

// Cartesian product of types for Sequential tests
using CudaForallReduceSanityTypes =
  Test< camp::cartesian_product<ReduceSanityDataTypeListBit, 
                                CudaResourceList, 
                                CudaForallExecPols,
                                CudaReducePols>>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Cuda,
                               ForallReduceSanityTestBit,
                               CudaForallReduceSanityTypes);

#endif
