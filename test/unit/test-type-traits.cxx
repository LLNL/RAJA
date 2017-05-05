#include "RAJA/RAJA.hpp"

#include "RAJA/internal/type_traits.hpp"

#include "gtest/gtest.h"

#ifdef RAJA_USE_OPENMP
static_assert(!RAJA::is_sequential_policy<RAJA::omp_parallel_for_exec>::value, "");
static_assert(RAJA::is_openmp_policy<RAJA::omp_parallel_for_exec>::value, "");
static_assert(!RAJA::is_cuda_policy<RAJA::omp_parallel_for_exec>::value, "");
#endif
static_assert(RAJA::is_sequential_policy<RAJA::seq_exec>::value, "");
static_assert(!RAJA::is_openmp_policy<RAJA::seq_exec>::value, "");
static_assert(!RAJA::is_cuda_policy<RAJA::seq_exec>::value, "");
#ifdef RAJA_USE_CUDA
static_assert(!RAJA::is_sequential_policy<RAJA::cuda_exec<128>>::value, "");
static_assert(!RAJA::is_openmp_policy<RAJA::cuda_exec<128>::value, "");
static_assert(RAJA::is_cuda_policy<RAJA::cuda_exec<128>>::value, "");
static_assert(!RAJA::is_sequential_policy<RAJA::cuda_exec_async<128>>::value, "");
static_assert(!RAJA::is_openmp_policy<RAJA::cuda_exec_async<128>::value, "");
static_assert(RAJA::is_cuda_policy<RAJA::cuda_exec_async<128>>::value, "");
#endif

TEST(TypeTraits, Default)
{
  ASSERT_EQ(true, true);
}
