//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "test-forall-atomic-basic.hpp"

#if defined(RAJA_ENABLE_OPENMP)
using OmpAtomicForallBasicTypes = 
  Test< camp::cartesian_product< OpenMPForallAtomicExecPols,
                                 OpenMPAtomicPols,
                                 HostResourceList,
                                 AtomicSegmentList,
                                 AtomicDataTypeList > >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P( OmpTest,
                                ForallAtomicBasicFunctionalTest,
                                OmpAtomicForallBasicTypes );
#endif
