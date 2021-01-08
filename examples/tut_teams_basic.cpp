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
 * Define host/device execution space in which 
 * to launch kernel. 
*/
using host_launch = RAJA::expt::seq_launch_t;
#if defined(RAJA_DEVICE_ACTIVE)
using device_launch = 

using exec_space = RAJA::expt:LaunchPolicy<
  RAJA::expt::seq_launch
>


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{



  return 0;
}
