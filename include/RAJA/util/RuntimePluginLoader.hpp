#ifndef RAJA_Runtime_Plugin_Loader_HPP
#define RAJA_Runtime_Plugin_Loader_HPP

#include "RAJA/util/PluginStrategy.hpp"
#include "RAJA/util/PluginOptions.hpp"

#include <vector>
#include <memory>

namespace RAJA
{
  namespace util
  {
    class RuntimePluginLoader : public ::RAJA::util::PluginStrategy
    {

    using Parent = ::RAJA::util::PluginStrategy;

    public:
      RuntimePluginLoader();

      void preLaunch(RAJA::util::PluginContext p);

      void postLaunch(RAJA::util::PluginContext p);

      void init(RAJA::util::PluginOptions p);

    private:
      // Initialize plugin from a shared object file specified by 'path'.
      void initPlugin(const std::string &path);
      void initDirectory(const std::string &path);
    
      std::vector<std::unique_ptr<Parent>> plugins;

    }; // end RuntimePluginLoader class
    
  } // end namespace util
} // end namespace RAJA

#endif
