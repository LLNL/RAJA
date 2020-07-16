//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_plugins_HPP
#define RAJA_plugins_HPP

#include "RAJA/util/PluginContext.hpp"
#include "RAJA/util/PluginStrategy.hpp"

namespace RAJA {
namespace util {

template <typename T>
auto trigger_updates_before(T&& item) -> typename std::remove_reference<T>::type
{
  return item;
}


inline
void
callPreCapturePlugins(PluginContext p) noexcept
{
  for (auto plugin = PluginRegistry::begin();
      plugin != PluginRegistry::end();
      ++plugin)
  {
    (*plugin).get()->preCapture(p);
  }
}

inline
void
callPostCapturePlugins(PluginContext p) noexcept
{
  for (auto plugin = PluginRegistry::begin();
      plugin != PluginRegistry::end();
      ++plugin)
  {
    (*plugin).get()->postCapture(p);
  }
}

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

} // closing brace for util namespace
} // closing brace for RAJA namespace

#endif
