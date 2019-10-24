//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
#ifndef RAJA_plugins_HPP
#define RAJA_plugins_HPP

#include "RAJA/util/PluginContext.hpp"
#include "RAJA/util/PluginStrategy.hpp"

namespace RAJA {
namespace util {

inline
void
callPreLaunchPlugins(PluginContext p)
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
callPostLaunchPlugins(PluginContext p)
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
