//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/util/PluginStrategy.hpp"

RAJA_INSTANTIATE_REGISTRY(PluginRegistry);

namespace RAJA {
namespace util {

PluginStrategy::PluginStrategy() = default;

void PluginStrategy::init(PluginOptions) { }

void PluginStrategy::preCapture(PluginContext) { }

void PluginStrategy::postCapture(PluginContext) { }

void PluginStrategy::preLaunch(PluginContext&) { }

void PluginStrategy::postLaunch(PluginContext&) { }

void PluginStrategy::finalize() { }

}
}
