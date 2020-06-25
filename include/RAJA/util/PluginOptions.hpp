#ifndef RAJA_Plugin_Options_HPP
#define RAJA_Plugin_Options_HPP

namespace RAJA {
namespace util {

#include <string>

struct PluginOptions
{
    PluginOptions(const std::string newstr) : str(newstr) {};
    
    std::string str;
};

PluginOptions make_options(const std::string& newstr)
{
    return PluginOptions{newstr};
}

} // namespace util
} // namespace RAJA

#endif
