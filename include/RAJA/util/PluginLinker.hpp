#ifndef RAJA_Plugin_Linker_HPP
#define RAJA_Plugin_Linker_HPP

#include "RAJA/util/RuntimePluginLoader.hpp"

namespace {
  struct pluginLinker {
    inline pluginLinker() { (void)RAJA::util::linkRuntimePluginLoader(); }
  } pluginLinker;
}
#endif