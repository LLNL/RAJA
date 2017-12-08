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

CUDA_TEST(Nested, CudaCollapse3)
{

  using Pol = RAJA::nested::Policy< 
    RAJA::nested::CudaCollapse<
    RAJA::nested::For<0, RAJA::cuda_threadblock_x_exec<16> >,
      RAJA::nested::For<1, RAJA::cuda_threadblock_y_exec<16> > > >;

  Index_type *sum1;
  cudaMallocManaged(&sum1, 1*sizeof(Index_type));
  
  Index_type *sum2;
  cudaMallocManaged(&sum2, 1*sizeof(Index_type));

  int N = 41;
  RAJA::nested::forall(Pol{},
                       camp::make_tuple(RAJA::RangeSegment(1, N),
                                        RAJA::RangeSegment(1, N)),
                       [=] RAJA_DEVICE (Index_type i, Index_type j) {
                         //printf("(%d, %d )\n", (int)i, (int) j );
                         
                         RAJA::atomic::atomicAdd<RAJA::atomic::cuda_atomic>(sum1,i);
                         RAJA::atomic::atomicAdd<RAJA::atomic::cuda_atomic>(sum2,j);

                       });
  
  cudaDeviceSynchronize();

  ASSERT_EQ( (N*(N-1)*(N-1))/2, *sum1);
  ASSERT_EQ( (N*(N-1)*(N-1))/2, *sum2);

  cudaFree(sum1);
  cudaFree(sum2);

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

CUDA_TEST(Nested, SubRange_ThreadBlock)
{
  using Pol = RAJA::nested::Policy<
      RAJA::nested::CudaCollapse<
        RAJA::nested::For<0, RAJA::cuda_threadblock_x_exec<1024>>
      > >;

  size_t num_elem = 2048;
  size_t first = 10;
  size_t last = num_elem - 10;

  double *ptr = nullptr;
  cudaErrchk(cudaMallocManaged(&ptr, sizeof(double) * num_elem) );

  RAJA::nested::forall(
      Pol{},
      camp::make_tuple(RAJA::RangeSegment(0, num_elem)),
      [=] RAJA_HOST_DEVICE (Index_type i) {
        ptr[i] = 0.0;
       });

  RAJA::nested::forall(
      Pol{},
      camp::make_tuple(RAJA::RangeSegment(first, last)),
      [=] RAJA_HOST_DEVICE (Index_type i) {
        ptr[i] = 1.0;
       });
  cudaDeviceSynchronize();

  size_t count = 0;
  for(size_t i = 0;i < num_elem; ++ i){
    count += ptr[i];
  }
  ASSERT_EQ(count, num_elem-20);
  for(size_t i = 0;i < 10;++ i){
    ASSERT_EQ(ptr[i], 0.0);
    ASSERT_EQ(ptr[num_elem-1-i], 0.0);
  }
}

CUDA_TEST(Nested, SubRange_Block)
{
  using Pol = RAJA::nested::Policy<
      RAJA::nested::CudaCollapse<
        RAJA::nested::For<0, RAJA::cuda_block_x_exec>
      > >;

  size_t num_elem = 2048;
  size_t first = 10;
  size_t last = num_elem - 10;

  double *ptr = nullptr;
  cudaErrchk(cudaMallocManaged(&ptr, sizeof(double) * num_elem) );

  RAJA::nested::forall(
      Pol{},
      camp::make_tuple(RAJA::RangeSegment(0, num_elem)),
      [=] RAJA_HOST_DEVICE (Index_type i) {
        ptr[i] = 0.0;
       });

  RAJA::nested::forall(
      Pol{},
      camp::make_tuple(RAJA::RangeSegment(first, last)),
      [=] RAJA_HOST_DEVICE (Index_type i) {
        ptr[i] = 1.0;
       });
  cudaDeviceSynchronize();

  size_t count = 0;
  for(size_t i = 0;i < num_elem; ++ i){
    count += ptr[i];
  }
  ASSERT_EQ(count, num_elem-20);
  for(size_t i = 0;i < 10;++ i){
    ASSERT_EQ(ptr[i], 0.0);
    ASSERT_EQ(ptr[num_elem-1-i], 0.0);
  }
}


CUDA_TEST(Nested, SubRange_Thread)
{
  using Pol = RAJA::nested::Policy<
      RAJA::nested::CudaCollapse<
        RAJA::nested::For<0, RAJA::cuda_thread_x_exec>
      > >;

  size_t num_elem = 1024;
  size_t first = 10;
  size_t last = num_elem - 10;

  double *ptr = nullptr;
  cudaErrchk(cudaMallocManaged(&ptr, sizeof(double) * num_elem) );

  RAJA::nested::forall(
      Pol{},
      camp::make_tuple(RAJA::RangeSegment(0, num_elem)),
      [=] RAJA_HOST_DEVICE (Index_type i) {
        ptr[i] = 0.0;
       });

  RAJA::nested::forall(
      Pol{},
      camp::make_tuple(RAJA::RangeSegment(first, last)),
      [=] RAJA_HOST_DEVICE (Index_type i) {
        ptr[i] = 1.0;
       });
  cudaDeviceSynchronize();

  size_t count = 0;
  for(size_t i = 0;i < num_elem; ++ i){
    count += ptr[i];
  }
  ASSERT_EQ(count, num_elem-20);
  for(size_t i = 0;i < 10;++ i){
    ASSERT_EQ(ptr[i], 0.0);
    ASSERT_EQ(ptr[num_elem-1-i], 0.0);
  }
}


CUDA_TEST(Nested, SubRange_Complex)
{
  using Pol = RAJA::nested::Policy<
      RAJA::nested::CudaCollapse<
        RAJA::nested::For<0, RAJA::cuda_thread_x_exec>
      > >;

  using ExecPolicy =
      RAJA::nested::Policy<
        RAJA::nested::CudaCollapse<
          RAJA::nested::For<0, RAJA::cuda_block_x_exec>,
          RAJA::nested::For<1, RAJA::cuda_thread_x_exec>>,
        RAJA::nested::For<2, RAJA::cuda_loop_exec> >;

  size_t num_elem = 1024;
  size_t first = 10;
  size_t last = num_elem - 10;

  double *ptr = nullptr;
  cudaErrchk(cudaMallocManaged(&ptr, sizeof(double) * num_elem) );

  RAJA::nested::forall(
      Pol{},
      camp::make_tuple(RAJA::RangeSegment(0, num_elem)),
      [=] RAJA_HOST_DEVICE (Index_type i) {
        ptr[i] = 0.0;
       });

  RAJA::nested::forall(
      ExecPolicy{},
      camp::make_tuple(RAJA::RangeSegment(first, last),
                       RAJA::RangeSegment(0, 16),
                       RAJA::RangeSegment(0, 32)),
      [=] RAJA_HOST_DEVICE (Index_type i, Index_type j, Index_type k) {
        //if(j == 0 && k == 0){
          RAJA::atomic::atomicAdd<RAJA::atomic::cuda_atomic>(ptr+i, 1.0);
        //}
       });
  cudaDeviceSynchronize();

  size_t count = 0;
  for(size_t i = 0;i < num_elem; ++ i){
    count += ptr[i];
  }
  ASSERT_EQ(count, (num_elem-20)*16*32);
  for(size_t i = 0;i < 10;++ i){
    ASSERT_EQ(ptr[i], 0.0);
    ASSERT_EQ(ptr[num_elem-1-i], 0.0);
  }
}


CUDA_TEST(Nested, SharedMemoryTestA)
{
  RAJA::SharedMemory<RAJA::cuda_shmem, double, 8> s;
  RAJA::SharedMemory<RAJA::cuda_shmem, float, 8> t;

  RAJA::forall<RAJA::cuda_exec<1024>>(RAJA::RangeSegment(0,8),
    [=] __device__ (int i){

      // Each thread assigns a value into shared memory
      printf("i=%d, s.data=%p, t.data=%p\n", i, &s[0], &t[0]);
      s[i] = i*1000.0;
      t[i] = 1.0/(i+1);

      __syncthreads();

      // Thread 0 prints all of the values, demonstrating the memory was shared
      if(i == 0){
        for(int j = 0;j < 8;++ j){
          printf("s[%d]=%lf, t[%d]=%f\n", j, s[j], j, t[j]);
        }
      }
  });

  // display printf's
  cudaDeviceSynchronize();
}

CUDA_TEST(Nested, SharedMemoryTestB)
{

  using pol =
      RAJA::nested::Policy<
        RAJA::nested::CudaCollapse<
          RAJA::nested::For<0, RAJA::cuda_thread_x_exec>,
          RAJA::nested::For<1, RAJA::cuda_thread_y_exec>,
        >
      >;

  RAJA::SharedMemory<RAJA::cuda_shmem, double, 4> s;
  RAJA::SharedMemory<RAJA::cuda_shmem, double, 16> t;

  RAJA::nested::forall(

    pol{},

    camp::make_tuple(RAJA::RangeSegment(0,4),
                     RAJA::RangeSegment(0,4)),

    [=] __device__ (int i, int j){

      printf("i=%d, j=%d, s.data=%p, t.data=%p\n", i, j, &s[0], &t[0]);

      // Clear s
      if(j == 0){
        s[i] = 0;
      }

      // Assign values to t
      t[i + 4*j] = i*j;
      printf("t[%d][%d] = %lf\n", i, j, t[i + 4*j]);


      __syncthreads();


      // Sum rows
      if(j == 0){
        for(int k = 0;k < 4; ++ k){
          s[i] += t[i + 4*k];
        }
      }



      __syncthreads();

      // Thread 0 prints all of the values of s
      if(i == 0 && j == 0){
        for(int k = 0;k < 4;++ k){
          printf("s[%d]=%lf\n", k, s[k]);
        }
      }
  });

  // display printf's
  cudaDeviceSynchronize();
}
#endif

#if defined(RAJA_ENABLE_OPENMP)
TEST(Nested, SharedMemoryTestC)
{

  using polI =
      RAJA::nested::Policy<
        RAJA::nested::For<0, RAJA::omp_for_nowait_exec>
      >;

  using polIJ =
      RAJA::nested::Policy<
        RAJA::nested::For<0, RAJA::omp_for_nowait_exec>,
        RAJA::nested::For<1, RAJA::loop_exec>
      >;

  RAJA::SharedMemory<RAJA::seq_shmem, double, 4> s;
  RAJA::SharedMemory<RAJA::seq_shmem, double, 16> t;

  double *output = new double[4];

  RAJA::nested::forall_multi(

      omp_multi_exec<true>{},

      // Zero out s[]
      RAJA::nested::makeLoop(
        polI{},
        camp::make_tuple(RAJA::RangeSegment(0,4)),
        [=] (int i){
          s[i] = 0;
        }),

      // Initialize t[]
      RAJA::nested::makeLoop(
        polIJ{},
        camp::make_tuple(RAJA::RangeSegment(0,4),
                         RAJA::RangeSegment(0,4)),
        [=] (int i, int j){
          t[i + 4*j] = i*j;
        }),

      // Compute s[] from t[]
      RAJA::nested::makeLoop(
        polI{},
        camp::make_tuple(RAJA::RangeSegment(0,4)),
        [=] (int i){
          for(int k = 0;k < 4;++ k){
            s[i] += t[i + 4*k];
          }
        }),

      // save output
      RAJA::nested::makeLoop(
        polI{},
        camp::make_tuple(RAJA::RangeSegment(0,4)),
        [=] (int i){
          output[i] = s[i];
        })

   );

  ASSERT_EQ(output[0], 0);
  ASSERT_EQ(output[1], 6);
  ASSERT_EQ(output[2], 12);
  ASSERT_EQ(output[3], 18);


  delete[] output;
}
#endif // RAJA_ENABLE_OPENMP


#if defined(RAJA_ENABLE_CUDA)
CUDA_TEST(Nested, SharedMemoryTestD)
{

  using polI = RAJA::nested::Policy<
      RAJA::nested::CudaCollapse<
        RAJA::nested::For<0, RAJA::cuda_thread_x_exec>
      > >;

  using polIJ = RAJA::nested::Policy<
      RAJA::nested::CudaCollapse<
        RAJA::nested::For<0, RAJA::cuda_thread_x_exec>,
        RAJA::nested::For<1, RAJA::cuda_thread_y_exec>
      > >;

  RAJA::SharedMemory<RAJA::cuda_shmem, double, 4> s;
  RAJA::SharedMemory<RAJA::cuda_shmem, double, 16> t;

  double *output = nullptr;
  cudaErrchk(cudaMallocManaged(&output, sizeof(double) * 4) );

  cudaDeviceSynchronize();

  RAJA::nested::forall_multi(

      cuda_multi_exec<false>{},

      // Zero out s[]
      RAJA::nested::makeLoop(
        polI{},
        camp::make_tuple(RAJA::RangeSegment(0,4)),
        [=] __device__ (int i){
          s[i] = 0;
        }),

      // Initialize t[]
      RAJA::nested::makeLoop(
        polIJ{},
        camp::make_tuple(RAJA::RangeSegment(0,4),
                         RAJA::RangeSegment(0,4)),
        [=] __device__ (int i, int j){
          t[i + 4*j] = i*j;
        }),

      // Compute s[] from t[]
      RAJA::nested::makeLoop(
        polI{},
        camp::make_tuple(RAJA::RangeSegment(0,4)),
        [=] __device__ (int i){
          for(int k = 0;k < 4;++ k){
            s[i] += t[i + 4*k];
          }
        }),

      // save output
      RAJA::nested::makeLoop(
        polI{},
        camp::make_tuple(RAJA::RangeSegment(0,4)),
        [=] __device__ (int i){
          output[i] = s[i];
        })

  );


  cudaDeviceSynchronize();

  ASSERT_EQ(output[0], 0);
  ASSERT_EQ(output[1], 6);
  ASSERT_EQ(output[2], 12);
  ASSERT_EQ(output[3], 18);


  cudaFree(&output);
}
#endif // RAJA_ENABLE_CUDA

