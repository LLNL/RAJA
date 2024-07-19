#if (__INTEL_CLANG_COMPILER && __INTEL_CLANG_COMPILER < 20230000)
// older version, use legacy header locations
#include <CL/sycl.hpp>
#else
// SYCL 2020 standard header
#include <sycl/sycl.hpp>
#endif
