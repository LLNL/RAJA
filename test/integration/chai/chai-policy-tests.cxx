#include "gtest/gtest.h"

#include "chai/ExecutionSpaces.hpp"

#include "RAJA/RAJA.hpp"

static_assert(RAJA::detail::get_space<RAJA::seq_exec>::value == chai::CPU, "");
static_assert(RAJA::detail::get_space<RAJA::IndexSet::ExecPolicy<RAJA::seq_segit, RAJA::seq_exec> >::value == chai::CPU, "");

#if defined(RAJA_ENABLE_OPENMP)
static_assert(RAJA::detail::get_space<RAJA::omp_parallel_for_exec>::value == chai::CPU, "");
#endif

#if defined(RAJA_ENABLE_CUDA)
static_assert(RAJA::detail::get_space<RAJA::cuda_exec<128> >::value == chai::GPU, "");
static_assert(RAJA::detail::get_space<RAJA::IndexSet::ExecPolicy<RAJA::seq_segit, RAJA::cuda_exec<128> > >::value == chai::GPU, "");
#endif

static_assert(RAJA::detail::get_space<RAJA::NestedPolicy< RAJA::ExecList< RAJA::seq_exec, RAJA::seq_exec > > >::value == chai::CPU, "");
static_assert(RAJA::detail::get_space<RAJA::NestedPolicy< RAJA::ExecList< RAJA::seq_exec, RAJA::cuda_exec<16> > > >::value == chai::GPU, "");

TEST(ChaiPolicyTest, Default) {
  std::cout << RAJA::detail::get_space<RAJA::IndexSet::ExecPolicy<RAJA::seq_segit, RAJA::cuda_exec<128> > >::value << std::endl;

  ASSERT_EQ(true, true);
}
