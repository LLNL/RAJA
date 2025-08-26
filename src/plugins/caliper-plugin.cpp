//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/util/CaliperPlugin.hpp"

#include <iostream>
#include <caliper/cali.h>

namespace RAJA
{
namespace util
{

void CaliperPlugin::preLaunch(const RAJA::util::PluginContext& p) override
{
  if (!p.kernel_name.empty())
  {
    CALI_MARK_BEGIN(p.kernel_name.c_str());
  }
}

void CaliperPlugin::postLaunch(const RAJA::util::PluginContext& p) override
{
  if (!p.kernel_name.empty())
  {
    CALI_MARK_END(p.kernel_name.c_str());
  }
}
};

void linkCaliperPlugin() {}

}  // namespace util
}  // namespace RAJA

// Dynamically loading plugin.
extern "C" RAJA::util::PluginStrategy* RAJAGetPlugin()
{
  return new CaliperPlugin;
}

// Statically loading plugin.
static RAJA::util::PluginRegistry::add<CaliperPlugin> P(
    "Caliper",
    "Enables Caliper Profiling");
