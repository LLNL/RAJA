#ifndef RAJA_KERNEL_NAME_HPP
#define RAJA_KERNEL_NAME_HPP

#include "RAJA/pattern/params/params_base.hpp"

namespace RAJA
{
namespace expt
{
namespace detail
{

struct KernelName : public ForallParamBase {
  RAJA_HOST_DEVICE KernelName() {}
  KernelName(const char* name_in) : name(name_in) {}
  const char* name;
};

}  // namespace detail

inline auto KernelName(const char* n) { return detail::KernelName(n); }
}  // namespace expt


}  //  namespace RAJA


#endif  // KERNEL_NAME_HPP
