#include "gtest/gtest.h"

#include "chai/ExecutionSpaces.hpp"

#include "RAJA/RAJA.hpp"

static_assert(RAJA::detail::get_space<RAJA::seq_exec>::value == chai::CPU, "");
#if defined(RAJA_ENABLE_OPENMP)
static_assert(RAJA::detail::get_space<RAJA::omp_parallel_for_exec>::value == chai::CPU, "");
#endif
#if defined(RAJA_ENABLE_CUDA)
static_assert(RAJA::detail::get_space<RAJA::cuda_exec<128> >::value == chai::GPU, "");
#endif

static_assert(RAJA::detail::get_space<RAJA::IndexSet::ExecPolicy<RAJA::seq_segit, RAJA::cuda_exec<128> > >::value == chai::NONE, "");

TEST(ChaiPolicyTest, Default) {
  std::cout << RAJA::detail::get_space<RAJA::IndexSet::ExecPolicy<RAJA::seq_segit, RAJA::cuda_exec<128> > >::value << std::endl;

  auto mp = RAJA::make_multi_policy<RAJA::seq_exec,RAJA::omp_parallel_for_exec>(
      [] (const RAJA::RangeSegment& r) {
        if (r.size() < 100) {
          return 0;
        } else {
          return 1;
        }
      });

  std::cout << RAJA::detail::get_space<decltype(mp)>::value << std::endl;


  ASSERT_EQ(true, true);
}


