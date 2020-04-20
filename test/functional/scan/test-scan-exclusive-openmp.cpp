//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-scan-exclusive.hpp"

#if defined(RAJA_ENABLE_OPENMP)

#include "../forall/test-forall-utils.hpp"
#include "../forall/test-forall-execpol.hpp"

using OpenMPExclusiveScanTypes = 
  Test< camp::cartesian_product< OpenMPForallExecPols,
                                 HostResourceList, 
                                 ScanOpTypes >>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(OpenMP, 
                               ScanFunctionalTest, 
                               OpenMPExclusiveScanTypes);

#endif
