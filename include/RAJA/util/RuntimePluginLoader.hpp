#ifndef RAJA_Runtime_Plugin_Loader_HPP
#define RAJA_Runtime_Plugin_Loader_HPP

#include "RAJA/util/PluginStrategy.hpp"
#include "RAJA/util/PluginOptions.hpp"

#include <dlfcn.h>
#include <dirent.h>
#include <vector>
#include <memory>

namespace RAJA
{
  namespace util
  {
    using Plugin = RAJA::util::PluginStrategy;

    class RuntimePluginLoader : public Plugin
    {
    public:
      RuntimePluginLoader()
      {
        char *env = ::getenv("RAJA_PLUGINS");
        if (nullptr == env)
        {
          return;
        }
        initDirectory(std::string(env));
      }

      void preLaunch(RAJA::util::PluginContext p)
      {
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

      void init(RAJA::util::PluginOptions p)
      {
        initDirectory(p.str);
      }

    private:
      // Initialize plugin from a shared object file specified by 'path'.
      void initPlugin(const std::string &path)
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

      // Initialize all plugins in a directory specified by 'path'.
      void initDirectory(const std::string &path)
      {
        DIR *dir;
        struct dirent *file;

        if ((dir = opendir(path.c_str())) != NULL)
        {
          while ((file = readdir(dir)) != NULL)
          {
            if (strcmp(file->d_name, ".") && strcmp(file->d_name, ".."))
            {
              initPlugin(path + "/" + file->d_name);
            }
          }
          closedir(dir);
        }
        else
        {
          perror("[PluginLoader]: Could not open plugin directory");
        }
      }

      std::vector<std::unique_ptr<Plugin>> plugins;

    }; // end RuntimePluginLoader class

    static RAJA::util::PluginRegistry::Add<RuntimePluginLoader> P("RuntimePluginLoader", "RuntimePluginLoader");

  } // end namespace util
} // end namespace RAJA

#endif
