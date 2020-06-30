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


class Plugin2CaptureStrategy
{
  public:
    Plugin2CaptureStrategy();

    virtual ~Plugin2CaptureStrategy() = default;

    virtual void preCapture(PluginContext p) = 0;

    virtual void postCapture(PluginContext p) = 0;
};

class Plugin2LaunchStrategy
{
  public:
    Plugin2LaunchStrategy();

    virtual ~Plugin2LaunchStrategy() = default;

    virtual void preLaunch(PluginContext p) = 0;

    virtual void postLaunch(PluginContext p) = 0;
};

using Plugin2CaptureRegistry = Registry<Plugin2CaptureStrategy>;
using Plugin2LaunchRegistry = Registry<Plugin2LaunchStrategy>;

} // closing brace for util namespace
} // closing brace for RAJA namespace


#endif
