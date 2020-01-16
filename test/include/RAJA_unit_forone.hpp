//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file of macro for-one CUDA unit tests.
/// Use as:
/// forone<<<1,1>>>( [=] __device__ () {} );
///

#if defined(RAJA_ENABLE_CUDA)
#include <RAJA/RAJA.hpp>

template <typename L>
__global__ void forone (L run)
{
  run();
}

#endif

