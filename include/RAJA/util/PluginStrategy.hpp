//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_PluginStrategy_HPP
#define RAJA_PluginStrategy_HPP

#include "RAJA/util/PluginContext.hpp"
#include "RAJA/util/PluginOptions.hpp"
#include "RAJA/util/Registry.hpp"

namespace RAJA {
namespace util {

class PluginStrategy
{
  public:
    PluginStrategy();

    virtual ~PluginStrategy() = default;

    virtual void init(const PluginOptions& p);

    virtual void preCapture(const PluginContext& p);

    virtual void postCapture(const PluginContext& p);

    virtual void preLaunch(const PluginContext& p);

    virtual void postLaunch(const PluginContext& p);

    virtual void finalize();
};

using PluginRegistry = Registry<PluginStrategy>;

} // closing brace for util namespace
} // closing brace for RAJA namespace


#endif
