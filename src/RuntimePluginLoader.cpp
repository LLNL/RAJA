//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/util/RuntimePluginLoader.hpp"

#include <dlfcn.h>
#include <dirent.h>

inline
bool
isSharedObject(const std::string& filename)
{
  return (filename.size() > 3 && !filename.compare(filename.size() - 3, 3, ".so"));
}

namespace RAJA {
namespace util {
  
RuntimePluginLoader::RuntimePluginLoader()
{
  char *env = ::getenv("RAJA_PLUGINS");
  if (nullptr == env)
  {
    return;
  }
  initDirectory(std::string(env));
}

void RuntimePluginLoader::init(RAJA::util::PluginOptions p)
{
  initDirectory(p.str);
  for (auto &plugin : plugins)
  {
    plugin->init(p);
  }
}

void RuntimePluginLoader::preCapture(RAJA::util::PluginContext p)
{
  for (auto &plugin : plugins)
  {
    plugin->preCapture(p);
  }
}

void RuntimePluginLoader::postCapture(RAJA::util::PluginContext p)
{
  for (auto &plugin : plugins)
  {
    plugin->postCapture(p);
  }
}

void RuntimePluginLoader::preLaunch(RAJA::util::PluginContext& p)
{
  for (auto &plugin : plugins)
  {
    plugin->preLaunch(p);
  }
}

void RuntimePluginLoader::postLaunch(RAJA::util::PluginContext& p)
{
  for (auto &plugin : plugins)
  {
    plugin->postLaunch(p);
  }
}

void RuntimePluginLoader::finalize()
{
  for (auto &plugin : plugins)
  {
    plugin->finalize();
  }
}


// Initialize plugin from a shared object file specified by 'path'.
void RuntimePluginLoader::initPlugin(const std::string &path)
{
  void *plugin = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!plugin)
  {
    printf("[RuntimePluginLoader]: dlopen failed: %s\n", dlerror());
  }

  RuntimePluginLoader::Parent *(*getPlugin)() = (RuntimePluginLoader::Parent * (*)()) dlsym(plugin, "getPlugin");

  if (getPlugin)
  {
    plugins.push_back(std::unique_ptr<RuntimePluginLoader::Parent>(getPlugin()));
  }
  else
  {
    printf("[RuntimePluginLoader]: dlsym failed: %s\n", dlerror());
  }
}

// Initialize all plugins in a directory specified by 'path'.
void RuntimePluginLoader::initDirectory(const std::string &path)
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
    perror("[RuntimePluginLoader]: Could not open plugin directory");
  }
}

void linkRuntimePluginLoader() {}

} // end namespace util
} // end namespace RAJA

static RAJA::util::PluginRegistry::add<RAJA::util::RuntimePluginLoader> P("RuntimePluginLoader", "Dynamically load RAJA plugins.");
