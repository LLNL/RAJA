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
#ifndef RAJA_PluginStrategy_HPP
#define RAJA_PluginStrategy_HPP

#include "RAJA/util/PluginContext.hpp"
#include "RAJA/util/Registry.hpp"

namespace RAJA {
namespace util {


class PluginStrategy 
{
  public:
    PluginStrategy();

    virtual ~PluginStrategy() = default;

    virtual void preLaunch(PluginContext p) = 0;

    virtual void postLaunch(PluginContext p) = 0;
};

using PluginRegistry = Registry<PluginStrategy>;

} // closing brace for util namespace
} // closing brace for RAJA namespace


#endif
