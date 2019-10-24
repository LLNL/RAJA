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
