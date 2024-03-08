//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
#include "RAJA/util/PluginStrategy.hpp"

#include <exception>

class ExceptionPlugin :
  public RAJA::util::PluginStrategy
{
  public:
  void preLaunch(const RAJA::util::PluginContext& RAJA_UNUSED_ARG(p)) override {
    throw std::runtime_error("preLaunch");
  }
};

extern "C" RAJA::util::PluginStrategy *getPlugin()
{
  return new ExceptionPlugin;
}
