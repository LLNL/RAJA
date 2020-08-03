//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/util/KokkosPluginLoader.hpp"

#include <dlfcn.h>
#include <dirent.h>

const uint64_t kokkos_interface_version = 20171029;

inline
bool
isSharedObject(const std::string& filename)
{
  return (filename.size() > 3 && !filename.compare(filename.size() - 3, 3, ".so"));
}

namespace RAJA {
namespace util {
  
  KokkosPluginLoader::KokkosPluginLoader()
  {
    char *env = ::getenv("KOKKOS_PLUGINS");
    if (env == nullptr)
    {
      return;
    }
    initDirectory(std::string(env));

    for (auto &func : init_functions)
    {
      func(0, kokkos_interface_version, 0, nullptr);
    }
  }

  void KokkosPluginLoader::preLaunch(RAJA::util::PluginContext& p)
  {
    for (auto &func : pre_functions)
    {
      func("", 0, &(p.kID));
    }
  }

  void KokkosPluginLoader::postLaunch(RAJA::util::PluginContext& p)
  {
    for (auto &func : post_functions)
    {
      func(p.kID);
    }
  }

  void KokkosPluginLoader::finalize()
  {
    for (auto &func : finalize_functions)
    {
      func();
    }
  }

  // Initialize plugin from a shared object file specified by 'path'.
  void KokkosPluginLoader::initPlugin(const std::string &path)
  {
    void *plugin = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!plugin)
    {
      printf("[KokkosPluginLoader]: dlopen failed: %s\n", dlerror());
    }

    // Getting and storing init function.
    init_function init = (init_function) dlsym(plugin, "kokkosp_init_library");
    if (init)
      init_functions.push_back(init);
    else
      printf("[KokkosPluginLoader]: dlsym failed: %s\n", dlerror());

    // Getting and storing preLaunch function.
    pre_function pre = (pre_function) dlsym(plugin, "kokkosp_begin_parallel_for");
    if (pre)
      pre_functions.push_back(pre);
    else
      printf("[KokkosPluginLoader]: dlsym failed: %s\n", dlerror());

    // Getting and storing postLaunch function.
    post_function post = (post_function) dlsym(plugin, "kokkosp_end_parallel_for");
    if (post)
      post_functions.push_back(post);
    else
      printf("[KokkosPluginLoader]: dlsym failed: %s\n", dlerror());

    // Getting and storing finalize function.
    finalize_function finalize = (finalize_function) dlsym(plugin, "kokkosp_finalize_library");
    if (post)
      finalize_functions.push_back(finalize);
    else
      printf("[KokkosPluginLoader]: dlsym failed: %s\n", dlerror());
  }

  // Initialize all plugins in a directory specified by 'path'.
  void KokkosPluginLoader::initDirectory(const std::string &path)
  {
    if (isSharedObject(path))
    {
      initPlugin(path);
      return;
    }
    
    DIR *dir;
    struct dirent *file;

    if ((dir = opendir(path.c_str())) != NULL)
    {
      while ((file = readdir(dir)) != NULL)
      {
        if (isSharedObject(std::string(file->d_name)))
        {
          initPlugin(path + "/" + file->d_name);
        }
      }
      closedir(dir);
    }
    else
    {
      perror("[KokkosPluginLoader]: Could not open plugin directory");
    }
  }

  void linkKokkosPluginLoader() {}

} // end namespace util
} // end namespace RAJA

static RAJA::util::PluginRegistry::add<RAJA::util::KokkosPluginLoader> P("KokkosPluginLoader", "Dynamically load plugins ported from the Kokkos library.");
