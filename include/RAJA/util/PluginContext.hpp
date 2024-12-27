//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_plugin_context_HPP
#define RAJA_plugin_context_HPP

#include <string>

#include "RAJA/policy/PolicyBase.hpp"
#include "RAJA/internal/get_platform.hpp"

namespace RAJA {
namespace util {

class KokkosPluginLoader;

struct PluginContext {
  public:
   PluginContext(const Platform p, const std::string *name = nullptr) :
   platform(p), kernel_name(name) {}

  Platform platform;
  const std::string *kernel_name;

  private:
    mutable uint64_t kID;

    friend class KokkosPluginLoader;
};

template<typename Policy>
PluginContext make_context(const std::string *name=nullptr)
{
  return PluginContext{detail::get_platform<Policy>::value, name};
}

} // closing brace for util namespace
} // closing brace for RAJA namespace

#endif
