//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

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
  using RAJA::at_v;
  using Pol = at_v<TypeParam, 0>;
  using IndexTypes = at_v<TypeParam, 1>;
  using Idx0 = at_v<IndexTypes, 0>;
  using Idx1 = at_v<IndexTypes, 1>;
  RAJA::ReduceSum<at_v<TypeParam, 2>, RAJA::Real_type> tsum(0.0);
  RAJA::Real_type total{0.0};
  auto ranges = RAJA::make_tuple(RAJA::TypedRangeSegment<Idx0>(0, x_len),
                                 RAJA::TypedRangeStrideSegment<Idx1>(0, y_len, 1));
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
using RAJA::list;
using s = RAJA::seq_exec;
using TestTypes =
    ::testing::Types<list<Policy<For<1, s>, For<0, s>>,
                          list<TypedIndex, Index_type>,
                          RAJA::seq_reduce>,
                     list<Policy<Tile<1, tile_fixed<2>, RAJA::loop_exec>,
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
        Policy<For<1, RAJA::omp_parallel_for_exec>, For<0, s>>,
        list<TypedIndex, Index_type>,
        RAJA::omp_reduce>,
    list<Policy<Tile<1, tile_fixed<2>, RAJA::omp_parallel_for_exec>,
                For<1, RAJA::loop_exec>,
                For<0, s>>,
         list<TypedIndex, Index_type>,
         RAJA::omp_reduce>>;
INSTANTIATE_TYPED_TEST_CASE_P(OpenMP, Nested, OMPTypes);
#endif
#if defined(RAJA_ENABLE_TBB)
using TBBTypes = ::testing::Types<
    list<Policy<For<1, RAJA::tbb_for_exec>, For<0, s>>,
         list<TypedIndex, Index_type>,
         RAJA::tbb_reduce>>;
INSTANTIATE_TYPED_TEST_CASE_P(TBB, Nested, TBBTypes);
#endif
#if defined(RAJA_ENABLE_CUDA)
using CUDATypes = ::testing::Types<
    list<Policy<For<1, s>, For<0, RAJA::cuda_exec<128>>>,
         list<TypedIndex, Index_type>,
         RAJA::cuda_reduce<128>>>;
INSTANTIATE_TYPED_TEST_CASE_P(CUDA, Nested, CUDATypes);
#endif

TEST(Nested, TileDynamic)
{
  RAJA::idx_t count = 0;
  RAJA::idx_t length = 5;
  RAJA::idx_t tile_size = 3;
  RAJA::nested::forall(
      RAJA::make_tuple(Tile<1, tile<2>, RAJA::seq_exec>{tile_size},
                       For<0, RAJA::seq_exec>{},
                       For<1, RAJA::seq_exec>{}),
      RAJA::make_tuple(RAJA::RangeSegment(0, length),
                       RAJA::RangeStrideSegment(0, length, 1)),
      [=, &count](Index_type i, Index_type j) {
//        std::cerr << "i: " << get_val(i) << " j: " << j << " count: " << count
//                  << std::endl;

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
      RAJA::make_tuple(RAJA::RangeSegment(0, 3),
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
      RAJA::make_tuple(RAJA::RangeSegment(0, 3),
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
                       RAJA::make_tuple(RAJA::RangeSegment(1, N),
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
      RAJA::make_tuple(RAJA::RangeSegment(0, 3),
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
      RAJA::make_tuple(RAJA::RangeSegment(0, 3),
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
      RAJA::make_tuple(RAJA::RangeSegment(0, 3),
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
      RAJA::make_tuple(RAJA::RangeSegment(0, num_elem)),
      [=] RAJA_HOST_DEVICE (Index_type i) {
        ptr[i] = 0.0;
       });

  RAJA::nested::forall(
      Pol{},
      RAJA::make_tuple(RAJA::RangeSegment(first, last)),
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
      RAJA::make_tuple(RAJA::RangeSegment(0, num_elem)),
      [=] RAJA_HOST_DEVICE (Index_type i) {
        ptr[i] = 0.0;
       });

  RAJA::nested::forall(
      Pol{},
      RAJA::make_tuple(RAJA::RangeSegment(first, last)),
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
      RAJA::make_tuple(RAJA::RangeSegment(0, num_elem)),
      [=] RAJA_HOST_DEVICE (Index_type i) {
        ptr[i] = 0.0;
       });

  RAJA::nested::forall(
      Pol{},
      RAJA::make_tuple(RAJA::RangeSegment(first, last)),
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
      RAJA::make_tuple(RAJA::RangeSegment(0, num_elem)),
      [=] RAJA_HOST_DEVICE (Index_type i) {
        ptr[i] = 0.0;
       });

  RAJA::nested::forall(
      ExecPolicy{},
      RAJA::make_tuple(RAJA::TypedRangeSegment<TypedIndex>(first, last),
                       RAJA::TypedRangeSegment<TypedIndex>(0, 16),
                       RAJA::TypedRangeSegment<TypedIndex>(0, 32)),
      [=] RAJA_HOST_DEVICE (TypedIndex i, TypedIndex j, TypedIndex k) {
        //if(j == 0 && k == 0){
          RAJA::atomic::atomicAdd<RAJA::atomic::cuda_atomic>(ptr+(*i), 1.0);
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

#endif

#ifdef RAJA_ENABLE_OPENMP
TEST(Nested, Collapse2)
{
  int N = 16;
  int M = 7;


  int *data = new int[N*M];
  for(int i = 0;i < M*N;++ i){
    data[i] = -1;
  }

  using Pol = RAJA::nested::Policy<
      RAJA::nested::OmpParallelCollapse<
        RAJA::nested::For<0>,
        RAJA::nested::For<1>
      > >;

  RAJA::nested::forall(
      Pol{},
      RAJA::make_tuple(
          RAJA::RangeSegment(0, N),
          RAJA::RangeSegment(0, M)),

      [=] (Index_type i, Index_type j) {
        data[i + j*N] = i;
       });

  for(int i = 0;i < N;++ i){
    for(int j = 0;j < M;++ j){
      ASSERT_EQ(data[i + j*N], i);
    }
  }


  delete[] data;
}


TEST(Nested, Collapse3)
{
  int N = 1;
  int M = 2;
  int K = 3;

  int *data = new int[N*M*K];
  for(int i = 0;i < M*N*K;++ i){
    data[i] = -1;
  }

  using Pol = RAJA::nested::Policy<
       RAJA::nested::OmpParallelCollapse<
       RAJA::nested::For<0>,
       RAJA::nested::For<1>,
       RAJA::nested::For<2>
        > >;

  RAJA::nested::forall(
        Pol{},
        RAJA::make_tuple(
        RAJA::RangeSegment(0, K),
        RAJA::RangeSegment(0, M),
        RAJA::RangeSegment(0, N) ),
        [=] (Index_type k, Index_type j, Index_type i) {
          data[i + N*(j + M*k)] = i + N*(j+M*k);
        });
  

  for(int k=0; k<K; k++){
    for(int j=0; j<M; ++j){
      for(int i=0; i<N; ++i){
        
        int id = i + N*(j + M*k);
        ASSERT_EQ(data[id], id);        
      }
    }
  }

  delete[] data;
}

TEST(Nested, Collapse4)
{
  int N = 1;
  int M = 2;
  int K = 3;

  int *data = new int[N*M*K];
  for(int i = 0;i < M*N*K;++ i){
    data[i] = -1;
  }

  using Pol = RAJA::nested::Policy<
       RAJA::nested::OmpParallelCollapse<
       RAJA::nested::For<0>,
       RAJA::nested::For<1>,
       RAJA::nested::For<2>
        > >;

  RAJA::nested::forall(
        Pol{},
        RAJA::make_tuple(
        RAJA::RangeSegment(0, K),
        RAJA::RangeSegment(0, M),
        RAJA::RangeSegment(0, N) ),
        [=] (Index_type k, Index_type j, Index_type i) {          
          Index_type  id = i + N * (j + M*k); 
          data[id] = id; 

        }); 

  for(int k=0; k<K; k++){
    for(int j=0; j<M; ++j){
      for(int i=0; i<N; ++i){
        
        int id = i + N*(j + M*k);
        ASSERT_EQ(data[id], id);        
      }
    }
  }

  delete[] data;
}


TEST(Nested, Collapse5)
{

  int N = 4;
  int M = 4;
  int K = 4;

  int *data = new int[N*M*K];
  for(int i = 0;i < M*N*K;++ i){
    data[i] = -1;
  }

  using Pol = RAJA::nested::Policy<
       RAJA::nested::OmpParallelCollapse<
       RAJA::nested::For<0>,
       RAJA::nested::For<1> >,
       RAJA::nested::For<2, RAJA::seq_exec> >; 

  RAJA::nested::forall(
        Pol{},
        RAJA::make_tuple(
        RAJA::RangeSegment(0, K),
        RAJA::RangeSegment(0, M),
        RAJA::RangeSegment(0, N) ),
        [=] (Index_type k, Index_type j, Index_type i) {

          data[i + N*(j + M*k)] = i + N*(j+M*k);
        });

  for(int k=0; k<K; ++k){
    for(int j=0; j<M; ++j){
      for(int i=0; i<N; ++i){
        
        int id = i + N*(j+M*k);
        ASSERT_EQ(data[id], id);
      }
    }
  }

  delete[] data;
}


TEST(Nested, Collapse6)
{

  int N = 3;
  int M = 3;
  int K = 4;

  int *data = new int[N*M];
  for(int i = 0; i< M*N; ++i){
    data[i] = 0;
  }

  using Pol = RAJA::nested::Policy<
       RAJA::nested::For<0, RAJA::seq_exec>, 
       RAJA::nested::OmpParallelCollapse<
       RAJA::nested::For<1>,
       RAJA::nested::For<2> > >;

  RAJA::nested::forall(
        Pol{},
        RAJA::make_tuple(
        RAJA::RangeSegment(0, K),
        RAJA::RangeSegment(0, M),
        RAJA::RangeSegment(0, N) ),
        [=] (Index_type k, Index_type j, Index_type i) {
          data[i + N*j] += k;
        });
  
  for(int j=0; j<M; ++j){
    for(int i=0; i<N; ++i){ 
      ASSERT_EQ(data[i +N*j], 6);
    }
  }


  delete[] data;
}

TEST(Nested, Collapse7)
{

  int N  = 3;
  int M  = 3;
  int K  = 4;
  int P  = 8;

  int *data = new int[N*M*K*P];
  for(int i = 0; i< N*M*K*P; ++i){
    data[i] = 0;
  }

  using Pol = RAJA::nested::Policy<
       RAJA::nested::For<0, RAJA::seq_exec>, 
       RAJA::nested::OmpParallelCollapse<
       RAJA::nested::For<1>,
       RAJA::nested::For<2>,
       RAJA::nested::For<3> > >;

  RAJA::nested::forall(
        Pol{},
        RAJA::make_tuple(
        RAJA::RangeSegment(0, K),
        RAJA::RangeSegment(0, M),
        RAJA::RangeSegment(0, N),
        RAJA::RangeSegment(0, P)
                         ),
        [=] (Index_type k, Index_type j, Index_type i, Index_type r) {
          Index_type id = r + P*(i + N*(j + M*k));
          data[id] += id;
        });

  for(int k=0; k<K; ++k){
    for(int j=0; j<M; ++j){
      for(int i=0; i<N; ++i){
        for(int r=0; r<P; ++r){
          Index_type id = r + P*(i + N*(j + M*k));
          ASSERT_EQ(data[id], id);
        }
      }
    }
  }

  delete[] data;
}


TEST(Nested, Collapse8)
{

  int N  = 3;
  int M  = 3;
  int K  = 4;
  int P  = 8;

  int *data = new int[N*M*K*P];
  for(int i = 0; i< N*M*K*P; ++i){
    data[i] = 0;
  }

  using Pol = RAJA::nested::Policy<
       RAJA::nested::OmpParallelCollapse<
       RAJA::nested::For<0>,
       RAJA::nested::For<1>,
       RAJA::nested::For<2> >,
       RAJA::nested::For<3, RAJA::seq_exec> >;

  RAJA::nested::forall(
        Pol{},
        RAJA::make_tuple(
        RAJA::RangeSegment(0, K),
        RAJA::RangeSegment(0, M),
        RAJA::RangeSegment(0, N),
        RAJA::RangeSegment(0, P)
                         ),
        [=] (Index_type k, Index_type j, Index_type i, Index_type r) {
          Index_type id = r + P*(i + N*(j + M*k));
          data[id] += id;
        });

  for(int k=0; k<K; ++k){
    for(int j=0; j<M; ++j){
      for(int i=0; i<N; ++i){
        for(int r=0; r<P; ++r){
          Index_type id = r + P*(i + N*(j + M*k));
          ASSERT_EQ(data[id], id);
        }
      }
    }
  }

  delete[] data;
}

#endif //RAJA_ENABLE_OPENMP
