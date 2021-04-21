//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Fundamental includes and structs used throughout RAJA tests.
//

#ifndef __RAJA_test_random_HPP__
#define __RAJA_test_random_HPP__

#include <random>

//
// Get random seed for tests
//
inline unsigned get_random_seed()
{
  static unsigned seed = std::random_device{}();
  return seed;
}


#endif // __RAJA_test_random_HPP__




