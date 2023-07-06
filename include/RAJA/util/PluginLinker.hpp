//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_Plugin_Linker_HPP
#define RAJA_Plugin_Linker_HPP

#include "RAJA/util/RuntimePluginLoader.hpp"
#include "RAJA/util/KokkosPluginLoader.hpp"

namespace {
  namespace anonymous_RAJA {
    struct pluginLinker {
      inline pluginLinker() {
        (void)RAJA::util::linkRuntimePluginLoader();
        (void)RAJA::util::linkKokkosPluginLoader();
      }
    } pluginLinker;
  }
}
#endif
