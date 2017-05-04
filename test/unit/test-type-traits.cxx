#include "RAJA/RAJA.hpp"

#include "RAJA/internal/type_traits.hpp"

#include "gtest/gtest.h"

static_assert(!RAJA::is_cuda_policy<RAJA::omp_parallel_for_exec>::value, "");
static_assert(RAJA::is_openmp_policy<RAJA::omp_parallel_for_exec>::value, "");
static_assert(!RAJA::is_cuda_policy<RAJA::seq_exec>::value, "");

TEST(TypeTraits, Default) 
{
  ASSERT_EQ(true, true);
}
