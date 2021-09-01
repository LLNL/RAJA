#ifndef KERNEL_NAME_HPP
#define KERNEL_NAME_HPP

namespace detail
{

  struct KernelName {
    RAJA_HOST_DEVICE KernelName() {}
    KernelName(const char* name_in) : name(name_in) {}
    const char* name;
    
    static constexpr size_t num_lambda_args = 0;
    RAJA_HOST_DEVICE auto get_lambda_arg_tup() { return camp::make_tuple(); }

  };

} // namespace detail

#include "cuda/kernel_name.hpp"

auto KernelName(const char * n)
{
  return detail::KernelName(n);
}


#endif // KERNEL_NAME_HPP
