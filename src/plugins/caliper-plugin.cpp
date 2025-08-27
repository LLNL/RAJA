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

CaliperPlugin::CaliperPlugin()
{
  std::cout<<"init caliper plugin"<<std::endl;
  const std::string varName = "RAJA_CALIPER";
  const char* val = std::getenv(varName.c_str());
  if (val == nullptr) {
    return;
  }
  
  ::RAJA::expt::detail::RAJA_caliper_profile = true;
}

void CaliperPlugin::preLaunch(const RAJA::util::PluginContext& p)
{
  std::cout<<"Calling prelaunch"<<std::endl;
  if (!p.kernel_name.empty())
  {
    CALI_MARK_BEGIN(p.kernel_name.c_str());
  }
}

void CaliperPlugin::postLaunch(const RAJA::util::PluginContext& p)
{
  if (!p.kernel_name.empty())
  {
    CALI_MARK_END(p.kernel_name.c_str());
  }
}

void linkCaliperPlugin() {}

}  // namespace util
}  // namespace RAJA

// Dynamically loading plugin.
//extern "C" RAJA::util::PluginStrategy* RAJAGetPlugin()
//{
//return new RAJA::util::CaliperPlugin;
//}

// Statically loading plugin.
static RAJA::util::PluginRegistry::add<RAJA::util::CaliperPlugin> P(
    "Caliper",
    "Enables Caliper Profiling");
