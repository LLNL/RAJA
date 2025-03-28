//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_plugin_context_HPP
#define RAJA_plugin_context_HPP

#include <string>

#include "RAJA/policy/PolicyBase.hpp"
#include "RAJA/internal/get_platform.hpp"

namespace RAJA
{
namespace util
{

class KokkosPluginLoader;

struct PluginContext
{
public:
  PluginContext(const Platform p, std::string&& name)
      : platform(p),
        kernel_name(std::move(name))
  {}

  Platform platform;
  std::string kernel_name;

private:
  mutable uint64_t kID;

  friend class KokkosPluginLoader;
};

template<typename Policy>
PluginContext make_context(std::string&& name)
{
  return PluginContext {detail::get_platform<Policy>::value, std::move(name)};
}

}  // namespace util
}  // namespace RAJA

#endif
