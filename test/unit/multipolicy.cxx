#include "gtest/gtest.h"

#include "RAJA/RAJA.hpp"

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
  RAJA::forall(mp, RAJA::RangeSegment(0, 5), [](RAJA::Index_type){
    ASSERT_EQ(omp_get_num_threads(), 1);
  });
  RAJA::forall(mp, RAJA::RangeSegment(0, 101), [](RAJA::Index_type){
    ASSERT_TRUE(omp_get_num_threads() > 1);
  });
  // Nest a multipolicy to ensure value-based policies are preserved
  auto mp2 = RAJA::make_multi_policy(
          std::make_tuple(RAJA::omp_parallel_for_exec{}, mp),
          [] (const RAJA::RangeSegment& r) {
        if (r.size() > 10 && r.size() < 90) {
          return 0;
        } else {
          return 1;
        }
      });
  RAJA::forall(mp2, RAJA::RangeSegment(0, 5), [](RAJA::Index_type){
    ASSERT_EQ(omp_get_num_threads(), 1);
  });
  RAJA::forall(mp2, RAJA::RangeSegment(0, 91), [](RAJA::Index_type){
    ASSERT_EQ(omp_get_num_threads(), 1);
  });
  RAJA::forall(mp2, RAJA::RangeSegment(0, 50), [](RAJA::Index_type){
    ASSERT_TRUE(omp_get_num_threads() > 1);
  });
}

