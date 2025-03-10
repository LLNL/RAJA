#ifndef RAJA_KERNEL_NAME_HPP
#define RAJA_KERNEL_NAME_HPP

#include "RAJA/pattern/params/params_base.hpp"

#define RAJA_KERNEL_NAME                                                       \
  RAJA::expt::detail::defaultKernelName(__func__, __FILE__, __LINE__)

namespace RAJA
{
namespace expt
{
namespace detail
{

inline const char* defaultKernelName(const char* const func_name,
                                     const char* const file_name,
                                     const int line_number)
{
  std::string defaultName;
  defaultName += std::string(func_name) + " " +
                 std::string(strrchr(file_name, '/')) + ":L" +
                 std::to_string(line_number);
  return defaultName.c_str();
}

struct KernelName : public ForallParamBase
{
  RAJA_HOST_DEVICE KernelName() {}

  KernelName(const char* name_in) : name(name_in) {}

  const char* name;
};

}  // namespace detail

inline auto KernelName(const char* n) { return detail::KernelName(n); }
}  // namespace expt


}  //  namespace RAJA


#endif  // KERNEL_NAME_HPP
