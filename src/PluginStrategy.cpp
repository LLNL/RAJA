//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/util/PluginStrategy.hpp"

RAJA_INSTANTIATE_REGISTRY(PluginRegistry);

namespace RAJA {
namespace util {

PluginStrategy::PluginStrategy() = default;

RAJASHAREDDLL_API void PluginStrategy::init(const PluginOptions& p) { }

RAJASHAREDDLL_API void PluginStrategy::preCapture(const PluginContext&, const RAJA::resources::Resource&) { }

RAJASHAREDDLL_API void PluginStrategy::postCapture(const PluginContext&, const RAJA::resources::Resource&) { }

RAJASHAREDDLL_API void PluginStrategy::preLaunch(const PluginContext&, const RAJA::resources::Resource&) { }

RAJASHAREDDLL_API void PluginStrategy::postLaunch(const PluginContext&, const RAJA::resources::Resource&) { }

RAJASHAREDDLL_API void PluginStrategy::finalize() { }

}
}
