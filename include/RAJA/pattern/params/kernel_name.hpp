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
    
    //static constexpr size_t num_lambda_args = 0;
    //RAJA_HOST_DEVICE auto get_lambda_arg_tup() { return camp::make_tuple(); }

  };

} // namespace detail

auto KernelName(const char * n)
{
  return detail::KernelName(n);
}
} // namespace expt


} //  namespace RAJA



#endif // KERNEL_NAME_HPP
