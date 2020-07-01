#ifndef RAJA_Plugin_Linker_HPP
#define RAJA_Plugin_Linker_HPP

#include "RAJA/util/RuntimePluginLoader.hpp"
#include "RAJA/util/KokkosPluginLoader.hpp"

namespace {
  struct pluginLinker {
    inline pluginLinker() {
      (void)RAJA::util::linkRuntimePluginLoader();
      (void)RAJA::util::linkKokkosPluginLoader();
    }
  } pluginLinker;
}
#endif