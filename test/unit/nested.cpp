#include "RAJA/RAJA.hpp"
#include "RAJA_gtest.hpp"
#include "RAJA/pattern/nested.hpp"

#include <cstdio>

RAJA_INDEX_VALUE(TypedIndex, "TypedIndex");
TEST(Nested, Basic)
{
  using namespace RAJA::nested;
  using Index_type = RAJA::Index_type;
  using pol =
      Policy<For<1, RAJA::seq_exec>, TypedFor<0, RAJA::seq_exec, TypedIndex>>;
  RAJA::nested::forall(pol{},
                       camp::make_tuple(RAJA::RangeSegment(0, 5),
                                        RAJA::RangeSegment(0, 5)),
                       [=](TypedIndex i, Index_type j) {
                         printf("%ld, %ld\n", *i, j);
                       });
}

TEST(Nested, collapse)
{
  using namespace RAJA::nested;
  using Index_type = RAJA::Index_type;
  using pol = Policy<Collapse<RAJA::seq_exec, For<0>, For<1>>>;
  RAJA::nested::forall(pol{},
                       camp::make_tuple(RAJA::RangeSegment(0, 5),
                                        RAJA::RangeSegment(0, 5)),
                       [=](Index_type i, Index_type j) {
                         printf("%ld, %ld\n", i, j);
                       });
}

#if defined(RAJA_ENABLE_CUDA)
#include <cuda_runtime.h>
CUDA_TEST(Nested, BasicCuda)
{
  using namespace RAJA::nested;
  using Index_type = RAJA::Index_type;
  using pol =
      Policy<For<1, RAJA::seq_exec>, TypedFor<0, RAJA::cuda_exec<128>, TypedIndex>>;
  RAJA::nested::forall(pol{},
                       camp::make_tuple(RAJA::RangeSegment(0, 5),
                                        RAJA::RangeSegment(0, 5)),
                       [=] __device__ (TypedIndex i, Index_type j) {
                         printf("%ld, %ld\n", *i, j);
                       });
}
#endif // RAJA_ENABLE_CUDA
