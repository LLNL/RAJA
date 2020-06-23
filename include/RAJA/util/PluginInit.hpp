#ifndef RAJA_Plugin_Init_HPP
#define RAJA_Plugin_Init_HPP

#include <string>
#include <vector>

namespace RAJA {
namespace plugin {

std::vector<std::string> paths;

void init(const std::string& path)
{   
    paths.push_back(path);
}

} // end namespace plugin
} // end namespace RAJA

#endif
