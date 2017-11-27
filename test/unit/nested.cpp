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


RAJA_HOST_DEVICE constexpr Index_type get_val(Index_type v) noexcept
{
  return v;
}
template <typename T>
RAJA_HOST_DEVICE constexpr Index_type get_val(T v) noexcept
{
  return *v;
}
CUDA_TYPED_TEST_P(Nested, Basic)
{
  using camp::at_v;
  using Pol = at_v<TypeParam, 0>;
  using IndexTypes = at_v<TypeParam, 1>;
  using Idx0 = at_v<IndexTypes, 0>;
  using Idx1 = at_v<IndexTypes, 1>;
  RAJA::ReduceSum<at_v<TypeParam, 2>, RAJA::Real_type> tsum(0.0);
  RAJA::Real_type total{0.0};
  auto ranges = camp::make_tuple(RAJA::RangeSegment(0, x_len),
                                 RAJA::RangeSegment(0, y_len));
  auto v = this->view;
  using namespace RAJA::nested;
  RAJA::nested::forall(Pol{}, ranges, [=] RAJA_HOST_DEVICE(Idx0 i, Idx1 j) {
    // std::cerr << "i: " << get_val(i) << " j: " << j << std::endl;
    v(get_val(i), j) = get_val(i) * x_len + j;
    tsum += get_val(i) * 1.1 + j;
  });
  for (Index_type i = 0; i < x_len; ++i) {
    for (Index_type j = 0; j < y_len; ++j) {
      ASSERT_EQ(this->view(i, j), i * x_len + j);
      total += i * 1.1 + j;
    }
  }
  ASSERT_FLOAT_EQ(total, tsum.get());
}

REGISTER_TYPED_TEST_CASE_P(Nested, Basic);

using namespace RAJA::nested;
using camp::list;
using s = RAJA::seq_exec;
using TestTypes =
    ::testing::Types<list<Policy<For<1, s>, TypedFor<0, s, TypedIndex>>,
                          list<TypedIndex, Index_type>,
                          RAJA::seq_reduce>,
                     list<Policy<Tile<1, tile_s<2>, RAJA::loop_exec>,
                                 Tile<0, tile<2>, RAJA::loop_exec>,
                                 For<0, s>,
                                 For<1, s>>,
                          list<Index_type, Index_type>,
                          RAJA::seq_reduce>,
                     list<Policy<Collapse<s, For<0>, For<1>>>,
                          list<Index_type, Index_type>,
                          RAJA::seq_reduce>>;

INSTANTIATE_TYPED_TEST_CASE_P(Sequential, Nested, TestTypes);

#if defined(RAJA_ENABLE_OPENMP)
using OMPTypes = ::testing::Types<
    list<
        Policy<For<1, RAJA::omp_parallel_for_exec>, TypedFor<0, s, TypedIndex>>,
        list<TypedIndex, Index_type>,
        RAJA::omp_reduce>,
    list<Policy<Tile<1, tile_s<2>, RAJA::omp_parallel_for_exec>,
                For<1, RAJA::loop_exec>,
                TypedFor<0, s, TypedIndex>>,
         list<TypedIndex, Index_type>,
         RAJA::omp_reduce>>;
INSTANTIATE_TYPED_TEST_CASE_P(OpenMP, Nested, OMPTypes);
#endif
#if defined(RAJA_ENABLE_TBB)
using TBBTypes = ::testing::Types<
    list<Policy<For<1, RAJA::tbb_for_exec>, TypedFor<0, s, TypedIndex>>,
         list<TypedIndex, Index_type>,
         RAJA::tbb_reduce>>;
INSTANTIATE_TYPED_TEST_CASE_P(TBB, Nested, TBBTypes);
#endif
#if defined(RAJA_ENABLE_CUDA)
using CUDATypes = ::testing::Types<
    list<Policy<For<1, s>, TypedFor<0, RAJA::cuda_exec<128>, TypedIndex>>,
         list<TypedIndex, Index_type>,
         RAJA::cuda_reduce<128>>>;
INSTANTIATE_TYPED_TEST_CASE_P(CUDA, Nested, CUDATypes);
#endif

TEST(Nested, TileDynamic)
{
  camp::idx_t count = 0;
  camp::idx_t length = 5;
  camp::idx_t tile_size = 3;
  RAJA::nested::forall(
      camp::make_tuple(Tile<1, tile<2>, RAJA::seq_exec>{tile_size},
                       For<0, RAJA::seq_exec>{},
                       For<1, RAJA::seq_exec>{}),
      camp::make_tuple(RAJA::RangeSegment(0, length),
                       RAJA::RangeSegment(0, length)),
      [=, &count](Index_type i, Index_type j) {
        std::cerr << "i: " << get_val(i) << " j: " << j << " count: " << count
                  << std::endl;

        ASSERT_EQ(count,
                  count < (length * tile_size)
                      ? (i * 3 + j)
                      : (length * tile_size)
                            + (i * (length - tile_size) + j - tile_size));
        count++;
      });
}


#if defined(RAJA_ENABLE_CUDA)
CUDA_TEST(Nested, CudaCollapse)
{

  using Pol = RAJA::nested::Policy<
      RAJA::nested::CudaCollapse<
        RAJA::nested::For<0, RAJA::cuda_thread_x_exec>,
        RAJA::nested::For<1, RAJA::cuda_threadblock_z_exec<4>>,
        RAJA::nested::For<2, RAJA::cuda_thread_y_exec> > >;

  RAJA::nested::forall(
      Pol{},
      camp::make_tuple(RAJA::RangeSegment(0, 3),
                       RAJA::RangeSegment(0, 2),
                       RAJA::RangeSegment(0, 5)),
      [=] RAJA_HOST_DEVICE (Index_type i, Index_type j, Index_type k) {
          printf("(%d, %d, %d)\n", (int)i, (int)j, (int)k);
       });
}


CUDA_TEST(Nested, CudaCollapse2)
{

  using Pol = RAJA::nested::Policy<
      RAJA::nested::CudaCollapse<
        RAJA::nested::For<0, RAJA::cuda_thread_x_exec>,
        RAJA::nested::For<1, RAJA::cuda_threadblock_z_exec<4>>
      >,
      RAJA::nested::For<2, RAJA::cuda_loop_exec> >;

  RAJA::nested::forall(
      Pol{},
      camp::make_tuple(RAJA::RangeSegment(0, 3),
                       RAJA::RangeSegment(0, 2),
                       RAJA::RangeSegment(0, 5)),
      [=] RAJA_DEVICE (Index_type i, Index_type j, Index_type k) {
          printf("(%d, %d, %d)\n", (int)i, (int)j, (int)k);
       });
}


CUDA_TEST(Nested, CudaReduceA)
{

  using Pol = RAJA::nested::Policy<
      RAJA::nested::CudaCollapse<
        RAJA::nested::For<0, RAJA::cuda_thread_x_exec>,
        RAJA::nested::For<1, RAJA::cuda_threadblock_z_exec<4>>
      >,
      RAJA::nested::For<2, RAJA::cuda_loop_exec> >;

  RAJA::ReduceSum<RAJA::cuda_reduce<12>, int> reducer(0);

  RAJA::nested::forall(
      Pol{},
      camp::make_tuple(RAJA::RangeSegment(0, 3),
                       RAJA::RangeSegment(0, 2),
                       RAJA::RangeSegment(0, 5)),
      [=] RAJA_DEVICE (Index_type i, Index_type j, Index_type k) {
        reducer += 1;
       });


  ASSERT_EQ((int)reducer, 3*2*5);
}


CUDA_TEST(Nested, CudaReduceB)
{

  using Pol = RAJA::nested::Policy<
      RAJA::nested::For<2, RAJA::loop_exec>,
      RAJA::nested::CudaCollapse<
        RAJA::nested::For<0, RAJA::cuda_thread_x_exec>,
        RAJA::nested::For<1, RAJA::cuda_threadblock_z_exec<4>>
      > >;

  RAJA::ReduceSum<RAJA::cuda_reduce<12>, int> reducer(0);

  RAJA::nested::forall(
      Pol{},
      camp::make_tuple(RAJA::RangeSegment(0, 3),
                       RAJA::RangeSegment(0, 2),
                       RAJA::RangeSegment(0, 5)),
      [=] RAJA_DEVICE (Index_type i, Index_type j, Index_type k) {
        reducer += 1;
       });


  ASSERT_EQ((int)reducer, 3*2*5);
}


CUDA_TEST(Nested, CudaReduceC)
{

  using Pol = RAJA::nested::Policy<
      RAJA::nested::For<2, RAJA::loop_exec>,
      RAJA::nested::For<0, RAJA::loop_exec>,
      RAJA::nested::For<1, RAJA::cuda_exec<12>>
      >;

  RAJA::ReduceSum<RAJA::cuda_reduce<12>, int> reducer(0);

  RAJA::nested::forall(
      Pol{},
      camp::make_tuple(RAJA::RangeSegment(0, 3),
                       RAJA::RangeSegment(0, 2),
                       RAJA::RangeSegment(0, 5)),
      [=] RAJA_DEVICE (Index_type i, Index_type j, Index_type k) {
        reducer += 1;
       });


  ASSERT_EQ((int)reducer, 3*2*5);
}

#endif
