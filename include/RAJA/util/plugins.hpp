//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_plugins_HPP
#define RAJA_plugins_HPP

#include "RAJA/util/PluginContext.hpp"
#include "RAJA/util/PluginOptions.hpp"
#include "RAJA/util/PluginStrategy.hpp"
#include "RAJA/util/RuntimePluginLoader.hpp"

namespace RAJA {
namespace util {

inline
void
callPreLaunchPlugins(PluginContext p) noexcept
{
  for (auto plugin = PluginRegistry::begin(); 
      plugin != PluginRegistry::end();
      ++plugin)
  {
    (*plugin).get()->preLaunch(p);
  }
}

inline
void
callPostLaunchPlugins(PluginContext p) noexcept
{
  for (auto plugin = PluginRegistry::begin(); 
      plugin != PluginRegistry::end();
      ++plugin)
  {
    (*plugin).get()->postLaunch(p);
  }
}

inline
void
callInitPlugins(PluginOptions p) noexcept
{
  for (auto plugin = PluginRegistry::begin(); 
      plugin != PluginRegistry::end();
      ++plugin)
  {
    (*plugin).get()->init(p);
  }
}

inline
void
init_plugins(const std::string& path)
{   
  callInitPlugins(make_options(path));
}

} // closing brace for util namespace
} // closing brace for RAJA namespace

#endif
