//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_PluginStrategy_HPP
#define RAJA_PluginStrategy_HPP

#include "RAJA/util/PluginContext.hpp"
#include "RAJA/util/PluginOptions.hpp"
#include "RAJA/util/Registry.hpp"
#include "RAJA/util/resource.hpp"
#include "camp/resource.hpp"

namespace RAJA {
namespace util {

class PluginStrategy
{
  public:
    RAJASHAREDDLL_API PluginStrategy();

    virtual ~PluginStrategy() = default;

    virtual RAJASHAREDDLL_API void init(const PluginOptions&);

    virtual RAJASHAREDDLL_API void preCapture(const PluginContext&, const resources::Resource&);

    virtual RAJASHAREDDLL_API void postCapture(const PluginContext&, const resources::Resource&);

    virtual RAJASHAREDDLL_API void preLaunch(const PluginContext&, const resources::Resource&);

    virtual RAJASHAREDDLL_API void postLaunch(const PluginContext&, const resources::Resource&);

    virtual RAJASHAREDDLL_API void finalize();
};

using PluginRegistry = Registry<PluginStrategy>;

} // closing brace for util namespace
} // closing brace for RAJA namespace


#endif
