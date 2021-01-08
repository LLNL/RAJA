//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "RAJA/RAJA.hpp"

/*
 *  Developing with RAJA Teams
 *
 *  This example serves as a basic overview of
 *  capabilities with the RAJA Teams API.
 *
 *  RAJA features shown:
 *    -  RAJA::expt::launch
 */

/*
 * The RAJA teams framework enables developers
 * to expressed algorithms in terms of nested
 * loops within an execution space. RAJA teams
 * enables run time selection of a host or
 * device execution space. As a starting point
 * the example below choses a sequential
 * execution space and either a CUDA or HIP
 * execution device execution space.
*/

// __host_launch_start
using host_launch = RAJA::expt::seq_launch_t;
// __host_launch_end

#if defined(RAJA_ENABLE_CUDA)
// __host_device_start
using device_launch = RAJA::expt::cuda_launch_t<false>;
// __host_device_end
#elif defined(RAJA_ENABLE_HIP)
using device_launch = RAJA::expt::cuda_launch_t<false>;
#endif

using exec_space = RAJA::expt::LaunchPolicy<
  host_launch
#if defined(RAJA_DEVICE_ACTIVE)
  ,device_launch
#endif
  >;


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{



  return 0;
}
