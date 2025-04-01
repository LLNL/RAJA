#ifndef RAJA_KERNEL_NAME_HPP
#define RAJA_KERNEL_NAME_HPP

#include "RAJA/pattern/params/params_base.hpp"

namespace RAJA
{
namespace detail
{

struct Name : public ::RAJA::expt::detail::ForallParamBase
{
  RAJA_HOST_DEVICE Name() {}

  Name(const char* name_in) : name(name_in) {}

  const char* name;
};

}  // namespace detail

inline auto Name(const char* n) { return detail::Name(n); }


}  //  namespace RAJA


#endif  // KERNEL_NAME_HPP
