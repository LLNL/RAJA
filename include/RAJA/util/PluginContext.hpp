//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
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

class KokkosPluginLoader;

struct PluginContext {
  public:
     PluginContext(const Platform p, const char *name_ = nullptr) :
       platform(p), name(name_) {}

    Platform platform;
    const char *name;   

  private:
    mutable uint64_t kID;

    friend class KokkosPluginLoader;
};


template<typename Policy>
PluginContext make_context()
{
  return PluginContext{detail::get_platform<Policy>::value};
}

} // closing brace for util namespace

template<typename Policy>
struct ExecContext : util::PluginContext {

  ExecContext(const char *name_ = nullptr) : 
    util::PluginContext{detail::get_platform<Policy>::value, name_}
  {

  }

};

} // closing brace for RAJA namespace

#endif
