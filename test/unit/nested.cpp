#include "RAJA/pattern/nested.hpp"
#include "RAJA/RAJA.hpp"
#include "RAJA_gtest.hpp"

#include <cstdio>

#if defined(RAJA_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

using RAJA::Index_type;
using RAJA::View;
using RAJA::Layout;
using layout_2d = Layout<2, RAJA::Index_type>;
using view_2d = View<Index_type, layout_2d>;
static constexpr Index_type x_len = 5;
static constexpr Index_type y_len = 5;

RAJA_INDEX_VALUE(TypedIndex, "TypedIndex");

template <typename NestedPolicy>
class Nested : public ::testing::Test
{
protected:
  Index_type* data;
  view_2d view{nullptr, x_len, y_len};

  virtual void SetUp()
  {
#if defined(RAJA_ENABLE_CUDA)
    cudaMallocManaged(&data,
                      sizeof(Index_type) * x_len * y_len,
                      cudaMemAttachGlobal);
#else
    data = new Index_type[x_len * y_len];
#endif
    view.set_data(data);
  }

  virtual void TearDown()
  {
#if defined(RAJA_ENABLE_CUDA)
    cudaFree(data);
#else
    delete[] data;
#endif
  }
};
TYPED_TEST_CASE_P(Nested);

constexpr Index_type get_val(Index_type v) { return v; }
template <typename T>
constexpr Index_type get_val(T v)
{
  return *v;
}
TYPED_TEST_P(Nested, Basic)
{
  using camp::at_v;
  using Pol = at_v<TypeParam, 0>;
  using IndexTypes = at_v<TypeParam, 1>;
  using Idx0 = at_v<IndexTypes, 0>;
  using Idx1 = at_v<IndexTypes, 1>;
  auto ranges = camp::make_tuple(RAJA::RangeSegment(0, x_len),
                                 RAJA::RangeSegment(0, y_len));
  using namespace RAJA::nested;
  RAJA::nested::forall(Pol{}, ranges, [=](Idx0 i, Idx1 j) {
    this->view(get_val(i), j) = get_val(i) * x_len + j;
  });
  for (Index_type i = 0; i < x_len; ++i) {
    for (Index_type j = 0; j < y_len; ++j) {
      ASSERT_EQ(this->view(i, j), i * x_len + j);
    }
  }
}

REGISTER_TYPED_TEST_CASE_P(Nested, Basic);

using namespace RAJA::nested;
using camp::list;
using s = RAJA::seq_exec;
using TestTypes = ::testing::Types<
    list<Policy<For<1, s>, TypedFor<0, s, TypedIndex>>,
         list<TypedIndex, Index_type>>,
    list<Policy<Collapse<s, For<0>, For<1>>>, list<Index_type, Index_type>>>;

INSTANTIATE_TYPED_TEST_CASE_P(Sequential, Nested, TestTypes);

#if defined(RAJA_ENABLE_OPENMP)
using OMPTypes =
    ::testing::Types<list<Policy<For<1, s>, TypedFor<0, s, TypedIndex>>,
                          list<TypedIndex, Index_type>>>;
INSTANTIATE_TYPED_TEST_CASE_P(OpenMP, Nested, OMPTypes);
#endif
#if defined(RAJA_ENABLE_CUDA)
using CUDATypes =
    ::testing::Types<list<Policy<For<1, s>, TypedFor<0, s, TypedIndex>>,
                          list<TypedIndex, Index_type>>>;
INSTANTIATE_TYPED_TEST_CASE_P(CUDA, Nested, CUDATypes);
#endif
