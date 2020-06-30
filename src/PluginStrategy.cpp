//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/util/PluginStrategy.hpp"

RAJA_DEFINE_REGISTRY();
RAJA_INSTANTIATE_REGISTRY(Plugin2CaptureRegistry);
RAJA_INSTANTIATE_REGISTRY(Plugin2LaunchRegistry);

namespace RAJA {
namespace util {

Plugin2CaptureStrategy::Plugin2CaptureStrategy() = default;
Plugin2LaunchStrategy::Plugin2LaunchStrategy() = default;

}
}
