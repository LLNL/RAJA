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

void PluginStrategy::init(const PluginOptions&) { }

void PluginStrategy::preCapture(const PluginContext&) { }

void PluginStrategy::postCapture(const PluginContext&) { }

void PluginStrategy::preLaunch(const PluginContext&) { }

void PluginStrategy::postLaunch(const PluginContext&) { }

void PluginStrategy::finalize() { }

}
}
