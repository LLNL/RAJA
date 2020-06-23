#ifndef RAJA_Runtime_Plugin_Loader_HPP
#define RAJA_Runtime_Plugin_Loader_HPP

#include "RAJA/util/PluginStrategy.hpp"
#include "RAJA/util/PluginInit.hpp"

#include <dlfcn.h>
#include <dirent.h>
#include <vector>
#include <memory>

using Plugin = RAJA::util::PluginStrategy;

class PluginLoader : public Plugin
{
public:
  PluginLoader()
  {
    char *env = ::getenv("RAJA_PLUGINS");
    if (nullptr == env)
    {
      return;
    }
    loadDirectory(env);
  }

  void preLaunch(RAJA::util::PluginContext p)
  {
    // Checking for new plugin directories to add.
    if (!RAJA::plugin::paths.empty())
    {
      for (auto &path : RAJA::plugin::paths)
      {
        loadDirectory(path.c_str());
      }
      RAJA::plugin::paths.clear();
    }

    for (auto &plugin : plugins)
    {
      plugin->preLaunch(p);
    }
  }

  void postLaunch(RAJA::util::PluginContext p)
  {
    for (auto &plugin : plugins)
    {
      plugin->postLaunch(p);
    }
  }

private:
  void loadDirectory(const char *env)
  {
    std::string path(env);
    DIR *dir;
    struct dirent *file;

    if ((dir = opendir(env)) != NULL)
    {
      while ((file = readdir(dir)) != NULL)
      {
        if (strcmp(file->d_name, ".") && strcmp(file->d_name, ".."))
        {
          loadPlugin(path + "/" + file->d_name);
        }
      }
      closedir(dir);
    }
    else
    {
      perror("[PluginLoader]: Could not open plugin directory");
    }
  }

  void loadPlugin(const std::string &path)
  {
    void *plugin = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!plugin)
    {
      printf("[PluginLoader]: Error: dlopen failed: %s\n", dlerror());
    }

    Plugin *(*getPlugin)() =
        (Plugin * (*)()) dlsym(plugin, "getPlugin");

    if (getPlugin)
    {
      plugins.push_back(std::unique_ptr<Plugin>(getPlugin()));
    }
    else
    {
      printf("Error: dlsym failed: %s\n", dlerror());
    }
  }

private:
  std::vector<std::unique_ptr<Plugin>> plugins;
};

static RAJA::util::PluginRegistry::Add<PluginLoader> P("RuntimePluginLoader", "RuntimePluginLoader");

#endif
