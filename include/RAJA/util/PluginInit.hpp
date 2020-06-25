#ifndef RAJA_Plugin_Init_HPP
#define RAJA_Plugin_Init_HPP

#include "RAJA/util/plugins.hpp"
#include "RAJA/util/PluginOptions.hpp"

#include <string>
#include <vector>

namespace RAJA {
namespace plugin {

void init(const std::string& path)
{   
    RAJA::util::callInitPlugins(RAJA::util::make_options(path));
}

} // end namespace plugin
} // end namespace RAJA

#endif
