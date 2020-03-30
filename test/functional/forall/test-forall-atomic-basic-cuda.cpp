//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing basic functional tests for atomic operations with forall.
///

#include "tests/test-forall-atomic-basic.hpp"

#if defined(RAJA_ENABLE_CUDA)
using AtomicCudaExecs = list< RAJA::cuda_exec<256> >;

using AtomicCudaPols = list< RAJA::auto_atomic,
                            RAJA::cuda_atomic
                           >;

using CudaAtomicForallBasicTypes = Test< cartesian_product<
                                          AtomicCudaExecs,
                                          AtomicCudaPols,
                                          CudaResourceList,
                                          AtomicDataTypes >
                                      >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P( CudaTest,
                                CudaForallAtomicBasicFunctionalTest,
                                CudaAtomicForallBasicTypes );
#endif
