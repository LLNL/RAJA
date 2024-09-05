//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_plugin_context_HPP
#define RAJA_plugin_context_HPP

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
  PluginContext(const Platform p) : platform(p) {}

  Platform platform;

private:
  mutable uint64_t kID;

  friend class KokkosPluginLoader;
};

template <typename Policy>
PluginContext make_context()
{
  return PluginContext{detail::get_platform<Policy>::value};
}

} // namespace util
} // namespace RAJA

#endif
