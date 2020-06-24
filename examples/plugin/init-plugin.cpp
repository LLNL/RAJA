#include "RAJA/util/PluginStrategy.hpp"
#include "RAJA/util/RuntimePluginLoader.hpp"

#include <iostream>
#include <chrono>

class InitPlugin : public RAJA::util::PluginStrategy
{
public:
    InitPlugin()
    {
        RAJA::plugin::initPlugin("./path/to/plugin");
        RAJA::plugin::initDirectory("./path/to/dir");
    }

    void preLaunch(RAJA::util::PluginContext RAJA_UNUSED_ARG(p))
    {
    }

    void postLaunch(RAJA::util::PluginContext RAJA_UNUSED_ARG(p))
    {
    }
};

// Factory function for RuntimePluginLoader.
extern "C" RAJA::util::PluginStrategy *getPlugin()
{
    return new InitPlugin;
}

static RAJA::util::PluginRegistry::Add<InitPlugin> P("init-plugin", "PluginInitializer");