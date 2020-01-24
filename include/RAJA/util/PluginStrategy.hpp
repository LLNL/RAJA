//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_PluginStrategy_HPP
#define RAJA_PluginStrategy_HPP

#include "RAJA/util/PluginContext.hpp"
#include "RAJA/util/Registry.hpp"

namespace RAJA {
namespace util {


class PluginStrategy 
{
  public:
    PluginStrategy();

    virtual ~PluginStrategy() = default;

    virtual void preLaunch(PluginContext p) = 0;

    virtual void postLaunch(PluginContext p) = 0;
};

using PluginRegistry = Registry<PluginStrategy>;

} // closing brace for util namespace
} // closing brace for RAJA namespace


#endif
