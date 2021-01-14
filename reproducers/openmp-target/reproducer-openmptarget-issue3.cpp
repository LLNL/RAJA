//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <random>

/*
 *  Reproducer for compile error when using random number generators
 *
 */

int main(int, char **)
{
  std::default_random_engine gen;
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  double dval = dist(gen);

  return (dval > 1.0 || dval < 0.0) ? 1 : 0;
}
