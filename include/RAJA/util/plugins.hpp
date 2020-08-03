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
#include "RAJA/util/KokkosPluginLoader.hpp"

namespace RAJA {
namespace util {

template <typename T>
RAJA_INLINE auto trigger_updates_before(T&& item)
  -> typename std::remove_reference<T>::type
{
  return item;
}


RAJA_INLINE
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

RAJA_INLINE
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

RAJA_INLINE
void
callPreLaunchPlugins(PluginContext& p)
{
  for (auto plugin = PluginRegistry::begin();
      plugin != PluginRegistry::end();
      ++plugin)
  {
    (*plugin).get()->preLaunch(p);
  }
}

RAJA_INLINE
void
callPostLaunchPlugins(PluginContext& p)
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
callInitPlugins(PluginOptions p)
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

inline
void
init_plugins()
{   
  callInitPlugins(make_options(""));
}


inline
void
finalize_plugins()
{   
  for (auto plugin = PluginRegistry::begin(); 
    plugin != PluginRegistry::end();
    ++plugin)
  {
    (*plugin).get()->finalize();
  }
}

} // closing brace for util namespace
} // closing brace for RAJA namespace

#endif
