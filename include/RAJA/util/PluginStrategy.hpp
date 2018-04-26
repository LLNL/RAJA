#ifndef RAJA_PluginStrategy_HPP
#define RAJA_PluginStrategy_HPP

#include "RAJA/util/Registry.hpp"

#include "RAJA/policy/PolicyBase.hpp"

namespace RAJA {
namespace util {


class PluginStrategy 
{
  public:
    PluginStrategy();

    virtual ~PluginStrategy() = default;

    virtual void preLaunch(Platform p) = 0;

    virtual void postLaunch(Platform p) = 0;
};

using PluginRegistry = Registry<PluginStrategy>;

}
}


#endif
