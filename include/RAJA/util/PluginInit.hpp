#ifndef RAJA_Plugin_Init_HPP
#define RAJA_Plugin_Init_HPP

namespace RAJA {
namespace plugin {

#include <string>

static std::string path;

void Init(const std::string& newpath)
{
    path = newpath;
}

} // end namespace plugin
} // end namespace RAJA

#endif