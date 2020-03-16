//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_plugin_context_HPP
#define RAJA_plugin_context_HPP

#include "RAJA/policy/PolicyBase.hpp"

#include "RAJA/internal/get_platform.hpp"

namespace RAJA {
namespace util {

struct PluginContext {
  PluginContext(const Platform p) :
    platform(p) {}

  Platform platform;
};

template<typename Policy>
PluginContext make_context()
{
  return PluginContext{detail::get_platform<Policy>::value};
}

} // closing brace for util namespace
} // closing brace for RAJA namespace

#endif
