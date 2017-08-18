#include "RAJA/pattern/nested.hpp"
#include "RAJA/RAJA.hpp"
#include "RAJA_gtest.hpp"

#include <cstdio>

using RAJA::Index_type;
using RAJA::View;
using RAJA::Layout;
using layout_2d = Layout<2, RAJA::Index_type>;
using view_2d = View<Index_type, layout_2d>;
static constexpr Index_type x_len = 5;
static constexpr Index_type y_len = 5;

RAJA_INDEX_VALUE(TypedIndex, "TypedIndex");
TEST(Nested, Basic)
{
  using namespace RAJA::nested;
  using pol =
      Policy<For<1, RAJA::seq_exec>, TypedFor<0, RAJA::seq_exec, TypedIndex>>;
  Index_type* data = new Index_type[x_len * y_len];
  view_2d v(data, x_len, y_len);
  RAJA::nested::forall(pol{},
                       camp::make_tuple(RAJA::RangeSegment(0, 5),
                                        RAJA::RangeSegment(0, 5)),
                       [=](TypedIndex i, Index_type j) {
                         v(*i, j) = (*i) * x_len + j;
                       });
  for (Index_type i = 0; i < x_len; ++i) {
    for (Index_type j = 0; j < y_len; ++j) {
      ASSERT_EQ(v(i, j), i * x_len + j);
    }
  }
  delete[] data;
}

TEST(Nested, collapse)
{
  using namespace RAJA::nested;
  using Index_type = RAJA::Index_type;
  using pol = Policy<Collapse<RAJA::seq_exec, For<0>, For<1>>>;
  Index_type* data = new Index_type[x_len * y_len];
  view_2d v(data, x_len, y_len);
  RAJA::nested::forall(pol{},
                       camp::make_tuple(RAJA::RangeSegment(0, 5),
                                        RAJA::RangeSegment(0, 5)),
                       [=](Index_type i, Index_type j) {
                         v(i, j) = i * x_len + j;
                       });
  for (Index_type i = 0; i < x_len; ++i) {
    for (Index_type j = 0; j < y_len; ++j) {
      ASSERT_EQ(v(i, j), i * x_len + j);
    }
  }
  delete[] data;
}
#if defined(RAJA_ENABLE_OPENMP)
TEST(Nested, BasicOMP)
{
  using namespace RAJA::nested;
  using Index_type = RAJA::Index_type;
  using pol = Policy<For<1, RAJA::seq_exec>,
                     TypedFor<0, RAJA::omp_parallel_for_exec, TypedIndex>>;
  Index_type* data = new Index_type[x_len * y_len];
  view_2d v(data, x_len, y_len);
  RAJA::nested::forall(pol{},
                       camp::make_tuple(RAJA::RangeSegment(0, 5),
                                        RAJA::RangeSegment(0, 5)),
                       [=](TypedIndex i, Index_type j) {
                         v(*i, j) = (*i) * x_len + j;
                       });
  for (Index_type i = 0; i < x_len; ++i) {
    for (Index_type j = 0; j < y_len; ++j) {
      ASSERT_EQ(v(i, j), i * x_len + j);
    }
  }
  delete[] data;
}
#endif  // RAJA_ENABLE_CUDA

#if defined(RAJA_ENABLE_CUDA)
#include <cuda_runtime.h>
CUDA_TEST(Nested, BasicCuda)
{
  using namespace RAJA::nested;
  using Index_type = RAJA::Index_type;
  using pol = Policy<For<1, RAJA::seq_exec>,
                     TypedFor<0, RAJA::cuda_exec<128>, TypedIndex>>;

  Index_type* data;
  cudaMallocManaged(&data,
                    sizeof(Index_type) * x_len * y_len,
                    cudaMemAttachGlobal);
  view_2d v(data, x_len, y_len);
  RAJA::nested::forall(pol{},
                       camp::make_tuple(RAJA::RangeSegment(0, 5),
                                        RAJA::RangeSegment(0, 5)),
                       [=] __device__(TypedIndex i, Index_type j) {
                         v(*i, j) = (*i) * x_len + j;
                       });
  for (Index_type i = 0; i < x_len; ++i) {
    for (Index_type j = 0; j < y_len; ++j) {
      ASSERT_EQ(v(i, j), i * x_len + j);
    }
  }
  cudaFree(data);
}
#endif  // RAJA_ENABLE_CUDA
