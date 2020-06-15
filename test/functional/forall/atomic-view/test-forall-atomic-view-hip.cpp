//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "test-forall-atomic-view.hpp"

#if defined(RAJA_ENABLE_HIP)
using HipAtomicForallViewTypes = 
  Test< camp::cartesian_product< HipForallExecPols,
                                 HipAtomicPols,
                                 HipResourceList,
                                 AtomicDataTypeList > >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P( HipTest,
                                ForallAtomicViewFunctionalTest,
                                HipAtomicForallViewTypes );
#endif
