#include "gtest/gtest.h"
#include "RAJA/RAJA.hxx"

TEST(multipolicy, basic)
{
  auto mp = RAJA::make_multi_policy<RAJA::seq_exec,RAJA::omp_parallel_for_exec>(
      [] (const RAJA::RangeSegment& r) {
        if (r.size() < 100) {
          return 0;
        } else {
          return 1;
        }
      });
  RAJA::forall(mp, RAJA::RangeSegment(0, 5), [](RAJA::Index_type i){
    ASSERT_EQ(omp_get_num_threads(), 1);
  });
  RAJA::forall(mp, RAJA::RangeSegment(0, 5), [](RAJA::Index_type i){
    ASSERT_TRUE(omp_get_num_threads() >= 1);
  });
}
