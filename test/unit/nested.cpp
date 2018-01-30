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
                                 RAJA::TypedRangeSegment<Idx1>(0, y_len));
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
    ::testing::Types<list<Policy<For<1, s, For<0, s, Lambda<0>>>>,
                          list<TypedIndex, Index_type>,
                          RAJA::seq_reduce>,
                     list<Policy<Tile<1, tile_fixed<2>, RAJA::loop_exec,
                                   Tile<0, tile_fixed<2>, RAJA::loop_exec,
                                     For<0, s,
                                       For<1, s, Lambda<0>>
                                     >
                                   >
                                 >>,
                          list<Index_type, Index_type>,
                          RAJA::seq_reduce>,
                     list<Policy<Collapse<s, ArgList<0,1>, Lambda<0>>>,
                          list<Index_type, Index_type>,
                          RAJA::seq_reduce>>;


INSTANTIATE_TYPED_TEST_CASE_P(Sequential, Nested, TestTypes);

#if defined(RAJA_ENABLE_OPENMP)
using OMPTypes = ::testing::Types<
    list<
        Policy<For<1, RAJA::omp_parallel_for_exec, For<0, s, Lambda<0>>>>,
        list<TypedIndex, Index_type>,
        RAJA::omp_reduce>,
    list<Policy<Tile<1, tile_fixed<2>, RAJA::omp_parallel_for_exec,
                For<1, RAJA::loop_exec,
                  For<0, s, Lambda<0>>
                >
               >>,
         list<TypedIndex, Index_type>,
         RAJA::omp_reduce>>;
INSTANTIATE_TYPED_TEST_CASE_P(OpenMP, Nested, OMPTypes);
#endif
#if defined(RAJA_ENABLE_TBB)
using TBBTypes = ::testing::Types<
    list<Policy<For<1, RAJA::tbb_for_exec, For<0, s, Lambda<0>>>>,
         list<TypedIndex, Index_type>,
         RAJA::tbb_reduce>>;
INSTANTIATE_TYPED_TEST_CASE_P(TBB, Nested, TBBTypes);
#endif
#if defined(RAJA_ENABLE_CUDA)
using CUDATypes = ::testing::Types<
    list<Policy<For<1, s, CudaKernel<For<0, RAJA::cuda_block_thread_exec<128>, Lambda<0>>>>>,
         list<TypedIndex, Index_type>,
         RAJA::cuda_reduce<1024>>>;
INSTANTIATE_TYPED_TEST_CASE_P(CUDA, Nested, CUDATypes);
#endif









#if defined(RAJA_ENABLE_CUDA)
CUDA_TEST(Nested, CudaCollapse1a)
{

  using Pol = RAJA::nested::Policy<
      CudaKernel<
        Collapse<RAJA::cuda_block_thread_exec<128>, ArgList<0,1,2>, Lambda<0>>>>;

  int *x = nullptr;
  cudaMallocManaged(&x, 3*2*5*sizeof(int));


  RAJA::nested::forall(
      Pol{},
      RAJA::make_tuple(RAJA::RangeSegment(0, 3),
                       RAJA::RangeSegment(0, 2),
                       RAJA::RangeSegment(0, 5)),
      [=] __device__ (Index_type i, Index_type j, Index_type k) {
        x[i + j*3 + k*3*2] = 1;
       });

  cudaDeviceSynchronize();

  for(int i = 0;i < 3*2*5;++ i){
    ASSERT_EQ(x[i], 1);
  }

  cudaFree(x);
}

CUDA_TEST(Nested, CudaCollapse1b)
{

  using Pol = RAJA::nested::Policy<
      CudaKernel<
        Collapse<RAJA::cuda_block_thread_exec<5>, ArgList<0,1>,
          For<2, RAJA::seq_exec, Lambda<0>>
        >
      >>;

  int *x = nullptr;
  cudaMallocManaged(&x, 3*2*5*sizeof(int));

  RAJA::nested::forall(
      Pol{},
      RAJA::make_tuple(RAJA::RangeSegment(0, 3),
                       RAJA::RangeSegment(0, 2),
                       RAJA::RangeSegment(0, 5)),
      [=] RAJA_DEVICE (Index_type i, Index_type j, Index_type k) {
        x[i + j*3 + k*3*2] = 1;
       });

  cudaDeviceSynchronize();

  for(int i = 0;i < 3*2*5;++ i){
    ASSERT_EQ(x[i], 1);
  }

  cudaFree(x);
}


//CUDA_TEST(Nested, CudaCollapse1c)
//{
//
//  using Pol = RAJA::nested::Policy<
//      CudaKernel<
//        Collapse<RAJA::cuda_block_seq_exec, ArgList<0,1>,
//          For<2, RAJA::cuda_thread_exec, Lambda<0>>
//        >
//      >>;
//
//  int *x = nullptr;
//  cudaMallocManaged(&x, 3*2*5*sizeof(int));
//
//  RAJA::nested::forall(
//      Pol{},
//      RAJA::make_tuple(RAJA::RangeSegment(0, 3),
//                       RAJA::RangeSegment(0, 2),
//                       RAJA::RangeSegment(0, 5)),
//      [=] RAJA_DEVICE (Index_type i, Index_type j, Index_type k) {
//        x[i + j*3 + k*3*2] = 1;
//       });
//
//  cudaDeviceSynchronize();
//
//  for(int i = 0;i < 3*2*5;++ i){
//    ASSERT_EQ(x[i], 1);
//  }
//
//  cudaFree(x);
//}





CUDA_TEST(Nested, CudaCollapse2)
{

  using Pol = RAJA::nested::Policy<
       CudaKernel<
         Collapse<RAJA::cuda_block_thread_exec<7>, ArgList<0,1>, Lambda<0>>
       >>;


  Index_type *sum1;
  cudaMallocManaged(&sum1, 1*sizeof(Index_type));

  Index_type *sum2;
  cudaMallocManaged(&sum2, 1*sizeof(Index_type));

  int N = 41;
  RAJA::nested::forall(Pol{},
                       RAJA::make_tuple(RAJA::RangeSegment(1, N),
                                        RAJA::RangeSegment(1, N)),
                       [=] RAJA_DEVICE (Index_type i, Index_type j) {

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
      CudaKernel<
        Collapse<RAJA::cuda_block_thread_exec<7>, ArgList<0,1>,
          For<2, RAJA::seq_exec, Lambda<0>>
        >
      >>;

  RAJA::ReduceSum<RAJA::cuda_reduce<1024>, int> reducer(0);

  RAJA::nested::forall(
      Pol{},
      RAJA::make_tuple(RAJA::RangeSegment(0, 3),
                       RAJA::RangeSegment(0, 2),
                       RAJA::RangeSegment(0, 5)),
      [=] RAJA_DEVICE (Index_type i, Index_type j, Index_type k) {
//        printf("b=%d,t=%d, i,j,k=%d,%d,%d\n",
//            (int)blockIdx.x, (int)threadIdx.x,
//            (int)i, (int)j, (int)k);
        reducer += 1;
       });


  ASSERT_EQ((int)reducer, 3*2*5);
}





CUDA_TEST(Nested, CudaReduceB)
{

  using Pol = RAJA::nested::Policy<
        For<2, RAJA::seq_exec,
          CudaKernel<
            Collapse<RAJA::cuda_block_thread_exec<7>, ArgList<0,1>, Lambda<0>>
          >
        >>;

  RAJA::ReduceSum<RAJA::cuda_reduce<1024>, int> reducer(0);

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
        For<2, RAJA::loop_exec,
          For<0, RAJA::loop_exec,
            CudaKernel<
              For<1, RAJA::cuda_block_thread_exec<45>, Lambda<0>>
            >
          >
        >>;

  RAJA::ReduceSum<RAJA::cuda_reduce<1024>, int> reducer(0);

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
        CudaKernel<
          For<0, RAJA::cuda_block_thread_exec<57>, Lambda<0>>
        >>;

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






//CUDA_TEST(Nested, SubRange_Complex)
//{
//  using PolA = RAJA::nested::Policy<
//          CudaKernel<
//            For<0, RAJA::cuda_block_thread_exec, Lambda<0>>
//          >>;
//
//  using PolB = RAJA::nested::Policy<
//          CudaKernel<
//            Collapse<RAJA::cuda_block_thread_exec, ArgList<0, 1>,
//              For<2, RAJA::seq_exec, Lambda<0>>
//            >
//          >>;
//
//
//  size_t num_elem = 1024;
//  size_t first = 10;
//  size_t last = num_elem - 10;
//
//  double *ptr = nullptr;
//  cudaErrchk(cudaMallocManaged(&ptr, sizeof(double) * num_elem) );
//
//  RAJA::nested::forall(
//      PolA{},
//      RAJA::make_tuple(RAJA::RangeSegment(0, num_elem)),
//      [=] RAJA_HOST_DEVICE (Index_type i) {
//        ptr[i] = 0.0;
//       });
//
//  RAJA::nested::forall(
//      PolB{},
//      RAJA::make_tuple(RAJA::RangeSegment(first, last),
//                       RAJA::RangeSegment(0, 16),
//                       RAJA::RangeSegment(0, 32)),
//      [=] RAJA_HOST_DEVICE (Index_type i, Index_type j, Index_type k) {
//        RAJA::atomic::atomicAdd<RAJA::atomic::cuda_atomic>(ptr+i, 1.0);
//       });
//
//
//  cudaDeviceSynchronize();
//
//  size_t count = 0;
//  for(size_t i = 0;i < num_elem; ++ i){
//    count += ptr[i];
//  }
//  ASSERT_EQ(count, (num_elem-20)*16*32);
//  for(size_t i = 0;i < 10;++ i){
//    ASSERT_EQ(ptr[i], 0.0);
//    ASSERT_EQ(ptr[num_elem-1-i], 0.0);
//  }
//}


#endif



TEST(Nested, Shmem1){
  using namespace RAJA;
  using namespace RAJA::nested;

  constexpr int TileSize = 3;
  using Pol = nested::Policy<
          nested::Tile<0, nested::tile_fixed<TileSize>, seq_exec,
            SetShmemWindow<
              For<0, seq_exec, Lambda<0>>,
              For<0, seq_exec, Lambda<1>>
            >
          >
        >;


  constexpr int N = 16;
  int *x = new int[N];
  for(int i = 0;i < N;++ i){
    x[i] = 0;
  }

  auto loop_segments = RAJA::make_tuple(RangeSegment(0,N));

  using shmem_t = SharedMemory<seq_shmem, int, TileSize>;
  ShmemWindowView<shmem_t, ArgList<0>, SizeList<TileSize>, decltype(loop_segments)> shmem;


  nested::forall(
      Pol{},

      loop_segments,

      [=](int i){
        shmem(i) = i;
      },
      [=](int i){
        x[i] = shmem(i) * 2;
      }
  );

  for(int i = 0;i < N;++ i){
    ASSERT_EQ(x[i], i*2);
  }

  delete[] x;
}


TEST(Nested, FissionFusion){
  using namespace RAJA;
  using namespace RAJA::nested;

  // Loop Fusion
  using Pol_Fusion = nested::Policy<
          For<0, seq_exec, Lambda<0>, Lambda<1>>
        >;

  // Loop Fission
  using Pol_Fission = nested::Policy<
          For<0, seq_exec, Lambda<0>>,
          For<0, seq_exec, Lambda<1>>
        >;


  constexpr int N = 16;
  int *x = new int[N];
  int *y = new int[N];
  for(int i = 0;i < N;++ i){
    x[i] = 0;
    y[i] = 0;
  }

  nested::forall(
      Pol_Fission{},

      RAJA::make_tuple(RangeSegment(0,N), RangeSegment(0,N)),

      [=](int i, int){
        x[i] += 1;
      },

      [=](int i, int){
        x[i] += 2;
      }
  );


  nested::forall(
      Pol_Fusion{},

      RAJA::make_tuple(RangeSegment(0,N), RangeSegment(0,N)),

      [=](int i, int){
        y[i] += 1;
      },

      [=](int i, int){
        y[i] += 2;
      }
  );

  for(int i = 0;i < N;++ i){
    ASSERT_EQ(x[i], y[i]);
  }

  delete[] x;
  delete[] y;
}


TEST(Nested, Tile){
  using namespace RAJA;
  using namespace RAJA::nested;

  // Loop Fusion
  using Pol = nested::Policy<
          nested::Tile<1, nested::tile_fixed<4>, seq_exec,
            For<0, seq_exec,
              For<1, seq_exec, Lambda<0>>
            >,
            For<0, seq_exec,
              For<1, seq_exec, Lambda<0>>
            >
          >,
          For<1, seq_exec, Lambda<1>>
        >;


  constexpr int N = 16;
  int *x = new int[N];
  for(int i = 0;i < N;++ i){
    x[i] = 0;
  }

  nested::forall(
      Pol{},

      RAJA::make_tuple(RangeSegment(0,N), RangeSegment(0,N)),

      [=](RAJA::Index_type i, RAJA::Index_type){
        x[i] += 1;
      },
      [=](RAJA::Index_type, RAJA::Index_type j){
        x[j] *= 10;
      }
  );

  for(int i = 0;i < N;++ i){
    ASSERT_EQ(x[i], 320);
  }

  delete[] x;
}


TEST(Nested, CollapseSeq){
  using namespace RAJA;
  using namespace RAJA::nested;

  // Loop Fusion
  using Pol = nested::Policy<
          Collapse<seq_exec, ArgList<0, 1>,
            Lambda<0>
          >
        >;


  constexpr int N = 16;
  int *x = new int[N*N];
  for(int i = 0;i < N*N;++ i){
    x[i] = 0;
  }

  nested::forall(
      Pol{},

      camp::make_tuple(RangeSegment(0,N), RangeSegment(0,N)),

      [=](RAJA::Index_type i, RAJA::Index_type j){
        x[i*N + j] += 1;
      }
  );

  for(int i = 0;i < N*N;++ i){
    ASSERT_EQ(x[i], 1);
  }

  delete[] x;
}

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
      RAJA::nested::Collapse<RAJA::nested::omp_parallel_collapse_exec, ArgList<0, 1>,
        Lambda<0>
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
      RAJA::nested::Collapse<RAJA::nested::omp_parallel_collapse_exec, ArgList<0, 1, 2>,
       Lambda<0>
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
        RAJA::nested::Collapse<RAJA::nested::omp_parallel_collapse_exec, ArgList<0, 1, 2>,
         Lambda<0>
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
      RAJA::nested::Collapse<RAJA::nested::omp_parallel_collapse_exec, ArgList<0, 1>,
        RAJA::nested::For<2, RAJA::seq_exec, Lambda<0> >
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
      RAJA::nested::For<0, RAJA::seq_exec,
        RAJA::nested::Collapse<RAJA::nested::omp_parallel_collapse_exec, ArgList<1, 2>,
          Lambda<0>
        >
      > >;


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
        RAJA::nested::For<0, RAJA::seq_exec,
          RAJA::nested::Collapse<RAJA::nested::omp_parallel_collapse_exec, ArgList<1, 2, 3>,
            Lambda<0>
          >
        > >;

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
        RAJA::nested::Collapse<RAJA::nested::omp_parallel_collapse_exec, ArgList<0, 1, 2>,
          RAJA::nested::For<3, RAJA::seq_exec, Lambda<0> >
          > >;

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

#ifdef RAJA_ENABLE_CUDA

//__global__ void myFoobar()
//{
//  int i =
//}

CUDA_TEST(Nested, CudaExec){
  using namespace RAJA;
  using namespace RAJA::nested;

  constexpr long N = 1024;

  // Loop Fusion
  using Pol = nested::Policy<
            CudaKernel<
              For<0, cuda_block_thread_exec<32>, Lambda<0>>
            >
        >;

//  double *d_ptr;
//  cudaErrchk(cudaMalloc(&d_ptr, sizeof(double) * N));


  RAJA::ReduceSum<cuda_reduce<1024>, long> trip_count(0);

  nested::forall(
      Pol{},

      RAJA::make_tuple(RangeSegment(0,N)),

      [=] __device__ (RAJA::Index_type i){

        trip_count += 1;
        //d_ptr[i] = 1;
        //d_ptr2[i] = 2;
      }
  );
  cudaDeviceSynchronize();

  long result = (long)trip_count;

  ASSERT_EQ(result, N);
}

CUDA_TEST(Nested, CudaExec1){
  using namespace RAJA;
  using namespace RAJA::nested;

  constexpr long N = (long)3*1024*1024;

  // Loop Fusion
  using Pol = nested::Policy<
            CudaKernel<
              For<0, cuda_block_thread_exec<32>, Lambda<0>>
            >
        >;


  RAJA::ReduceSum<cuda_reduce<128>, long> trip_count(0);

  nested::forall(
      Pol{},

      RAJA::make_tuple(RangeSegment(0,N)),

      [=] __device__ (ptrdiff_t i){

        trip_count += 1;
      }
  );
  cudaDeviceSynchronize();

  long result = (long)trip_count;
  //printf("result=%ld\n", result);

  ASSERT_EQ(result, N);
}


CUDA_TEST(Nested, CudaExec1a){
  using namespace RAJA;
  using namespace RAJA::nested;

  constexpr long N = (long)128;

  // Loop Fusion
  using Pol = nested::Policy<
            CudaKernel<
              nested::Collapse<cuda_thread_exec, ArgList<0,1,2>, Lambda<0>>
            >
        >;


  RAJA::ReduceSum<cuda_reduce<1024>, long> trip_count(0);

  nested::forall(
      Pol{},

      RAJA::make_tuple(RangeSegment(0,N), RangeSegment(0,N), RangeStrideSegment(0,N,2)),

      [=] __device__ (ptrdiff_t i, ptrdiff_t j, ptrdiff_t k){
        trip_count += 1;
      }
  );
  cudaDeviceSynchronize();

  long result = (long)trip_count;
  //printf("result=%ld\n", result);

  ASSERT_EQ(result, N*N*N/2);
}


CUDA_TEST(Nested, CudaExec1ab){
  using namespace RAJA;
  using namespace RAJA::nested;

  constexpr long N = (long)128;

  // Loop Fusion
  using Pol = nested::Policy<
            CudaKernel<
              nested::Collapse<cuda_block_exec, ArgList<0,1,2>, Lambda<0>>
            >
        >;


  RAJA::ReduceSum<cuda_reduce<1024>, long> trip_count(0);

  nested::forall(
      Pol{},

      RAJA::make_tuple(RangeSegment(0,N), RangeSegment(0,N), RangeStrideSegment(0,N,2)),

      [=] __device__ (ptrdiff_t i, ptrdiff_t j, ptrdiff_t k){
        trip_count += 1;
      }
  );
  cudaDeviceSynchronize();

  long result = (long)trip_count;
  //printf("result=%ld\n", result);

  ASSERT_EQ(result, N*N*N/2);
}


CUDA_TEST(Nested, CudaExec1ac){
  using namespace RAJA;
  using namespace RAJA::nested;

  constexpr long N = (long)128;

  // Loop Fusion
  using Pol = nested::Policy<
            CudaKernel<
              nested::Collapse<cuda_block_thread_exec<1024>, ArgList<0,1,2>, Lambda<0>>
            >
        >;


  RAJA::ReduceSum<cuda_reduce<1024>, long> trip_count(0);

  nested::forall(
      Pol{},

      RAJA::make_tuple(RangeSegment(0,N), RangeSegment(0,N), RangeStrideSegment(0,N,2)),

      [=] __device__ (ptrdiff_t i, ptrdiff_t j, ptrdiff_t k){
        trip_count += 1;
      }
  );
  cudaDeviceSynchronize();

  long result = (long)trip_count;
  //printf("result=%ld\n", result);

  ASSERT_EQ(result, N*N*N/2);
}


CUDA_TEST(Nested, CudaExec1b){
  using namespace RAJA;
  using namespace RAJA::nested;

  constexpr long N = (long)3*1024*1024;

  // Loop Fusion
  using Pol = nested::Policy<
            CudaKernel<
              For<0, cuda_block_thread_exec<128>, Lambda<0>>
              >
        >;


  RAJA::ReduceSum<cuda_reduce<1024>, long> trip_count(0);

  nested::forall(
      Pol{},

      RAJA::make_tuple(RangeSegment(0,N)),

      [=] __device__ (ptrdiff_t i){

        trip_count += 1;
        //printf("[%d] %d\n", (int)threadIdx.x, (int)i);
      }
  );
  cudaDeviceSynchronize();

  long result = (long)trip_count;
  //printf("result=%ld\n", result);

  ASSERT_EQ(result, N);
}




CUDA_TEST(Nested, CudaExec1c){
  using namespace RAJA;
  using namespace RAJA::nested;

  constexpr long N = (long)3;

  // Loop Fusion
  using Pol = nested::Policy<
      CudaKernelBase<cuda_explicit_launch<false, 5, 3>,
            //CudaKernel<
              For<0, cuda_block_exec,
                For<1, cuda_block_exec,
                  For<2, cuda_block_thread_exec<2>, Lambda<0>>
                >
              >
            >
        >;


  RAJA::ReduceSum<cuda_reduce<1024>, long> trip_count(0);

  nested::forall(
      Pol{},

      RAJA::make_tuple(RangeSegment(0,N), RangeSegment(0,N), RangeSegment(0,N)),

      [=] __device__ (RAJA::Index_type i, RAJA::Index_type j, RAJA::Index_type k){

        trip_count += 1;
        //printf("[%d,%d] %d,%d,%d\n", (int)blockIdx.x, (int)threadIdx.x, (int)i, (int)j, (int)k);
      }
  );
  cudaDeviceSynchronize();

  long result = (long)trip_count;
  //printf("result=%ld\n", result);

  ASSERT_EQ(result, N*N*N);
}








CUDA_TEST(Nested, CudaComplexNested){
  using namespace RAJA;
  using namespace RAJA::nested;

  constexpr long N = (long)739;

  using Pol = nested::Policy<
            CudaKernel<
              For<0, cuda_block_thread_exec<15>,
                For<1, cuda_thread_exec,
                  For<2, cuda_thread_exec, Lambda<0>>
                >,
                For<2, cuda_thread_exec, Lambda<0>>
              >
            >
          >;

  int *ptr = nullptr;
  cudaErrchk(cudaMallocManaged(&ptr, sizeof(int) * N) );

  for(long i = 0;i < N;++ i){
    ptr[i] = 0;
  }


  auto segments = RAJA::make_tuple(RangeSegment(0,N), RangeSegment(0,N), RangeSegment(0, N));


  RAJA::ReduceSum<cuda_reduce<1024>, long> trip_count(0);

  nested::forall(
      Pol{},

      segments,

      [=] __device__ (RAJA::Index_type i, RAJA::Index_type j, RAJA::Index_type k){

        trip_count += 1;
        RAJA::atomic::atomicAdd<RAJA::atomic::auto_atomic>(ptr+i, (int)1);

      }
  );
  cudaDeviceSynchronize();

  for(long i = 0;i < N;++ i){
    ASSERT_EQ(ptr[i], (int)(N*N + N));
  }

  // check trip count
  long result = (long)trip_count;
  ASSERT_EQ(result, N*N*N + N*N);


  cudaFree(ptr);
}




CUDA_TEST(Nested, CudaShmemWindow1d){
  using namespace RAJA;
  using namespace RAJA::nested;

  constexpr long N = (long)256;

  using Pol = nested::Policy<
            CudaKernel<
              nested::Tile<0, nested::tile_fixed<16>, seq_exec,
                SetShmemWindow<

                  For<0, cuda_thread_exec, Lambda<0>>,

                  CudaThreadSync,

                  For<0, cuda_thread_exec, Lambda<1>>
                > // SetShmemWindow
              >
            >
          >;

  int *ptr = nullptr;
  cudaErrchk(cudaMallocManaged(&ptr, sizeof(int) * N) );

  for(long i = 0;i < N;++ i){
    ptr[i] = 0;
  }


  auto segments = RAJA::make_tuple(RangeSegment(0,N));


  RAJA::ReduceSum<cuda_reduce<1024>, long> trip_count(0);


  using shmem_t = SharedMemory<cuda_shmem, double, 16>;
  ShmemWindowView<shmem_t, ArgList<0>, SizeList<16>, decltype(segments)> shmem;


  nested::forall(
      Pol{},

      segments,

      [=] __device__ (RAJA::Index_type i){

        trip_count += 1;
        shmem(i) = i;

      },

      [=] __device__ (RAJA::Index_type i){

        trip_count += 1;
        ptr[i] = shmem(i);

      }
  );
  cudaDeviceSynchronize();

  for(long i = 0;i < N;++ i){
    ASSERT_EQ(ptr[i], (int)(i));
  }

  // check trip count
  long result = (long)trip_count;
  ASSERT_EQ(result, 2*N);


  cudaFree(ptr);
}



CUDA_TEST(Nested, CudaShmemWindow2d){
  using namespace RAJA;
  using namespace RAJA::nested;

  constexpr long N = (long)7;
  constexpr long M = (long)5;

  using Pol = nested::Policy<
            CudaKernel<
              nested::Tile<0, nested::tile_fixed<2>, seq_exec,
                nested::Tile<1, nested::tile_fixed<2>, seq_exec,
                  SetShmemWindow<
                    For<0, cuda_thread_exec,

                      For<1, cuda_thread_exec, Lambda<0>>,

                      CudaThreadSync,

                      For<1, cuda_thread_exec, Lambda<1>>
                    >
                  >
                >
              >
            >
          >;

  int *ptr = nullptr;
  cudaErrchk(cudaMallocManaged(&ptr, sizeof(int) * N * M) );

  for(long i = 0;i < N*M;++ i){
    ptr[i] = 0;
  }


  auto segments = RAJA::make_tuple(RangeSegment(0,N), RangeSegment(0,M));


  RAJA::ReduceSum<cuda_reduce<512>, long> trip_count(0);


  using shmem_t = SharedMemory<cuda_shmem, double, 4>;
  ShmemWindowView<shmem_t, ArgList<0,1>, SizeList<2,2>, decltype(segments)> shmem;
  ShmemWindowView<shmem_t, ArgList<0,1>, SizeList<2,2>, decltype(segments)> shmem2;


  nested::forall(
      Pol{},

      segments,

      [=] __device__ (RAJA::Index_type i, RAJA::Index_type j){
        trip_count += 1;
        shmem(i,j) = i*j;
        shmem2(i,j) = 2*i*j;
      },

      [=] __device__ (RAJA::Index_type i, RAJA::Index_type j){

        trip_count += 1;
        //ptr[i*M + j] = shmem(i,j);
        ptr[i*M + j] = shmem(i,j) + shmem2(i,j);

      }
  );
  cudaDeviceSynchronize();

  for(long i = 0;i < N;++ i){
    for(long j = 0;j < M;++ j){
      ASSERT_EQ(ptr[i*M+j], (int)(3*i*j));
      //ASSERT_EQ(ptr[i*M+j], (int)(i*j));
    }
  }

  // check trip count
  long result = (long)trip_count;
  ASSERT_EQ(result, 2*N*M);


  cudaFree(ptr);
}




#endif // CUDA



#ifdef RAJA_ENABLE_CUDA

CUDA_TEST(Nested, CudaExec_1threadexec){
  using namespace RAJA;
  using namespace RAJA::nested;

  constexpr long N = (long)1024*1024;

  // Loop Fusion
  using Pol = nested::Policy<
            CudaKernel<
              For<0, cuda_thread_exec, Lambda<0>>,
              For<0, cuda_thread_exec, Lambda<0>>,
              For<0, cuda_thread_exec, Lambda<0>>,
              For<0, cuda_thread_exec, Lambda<0>>
            >
        >;


  RAJA::ReduceSum<cuda_reduce<1024>, long> trip_count(0);

  nested::forall(
      Pol{},

      RAJA::make_tuple(RangeSegment(0,N)),

      [=] __device__ (ptrdiff_t i){

        if(i > -1){
          trip_count += 1;
        }

      }
  );
  cudaDeviceSynchronize();


  long result = (long)trip_count;
  //printf("result=%ld\n", result);

  ASSERT_EQ(result, 4*N);
}

CUDA_TEST(Nested, CudaExec_1blockexec){
  using namespace RAJA;
  using namespace RAJA::nested;

  constexpr long N = (long)1024*1024;

  // Loop Fusion
  using Pol = nested::Policy<
            CudaKernel<
              For<0, cuda_block_exec, Lambda<0>>
            >
        >;


  RAJA::ReduceSum<cuda_reduce<1024>, long> trip_count(0);

  nested::forall(
      Pol{},

      RAJA::make_tuple(RangeSegment(0,N)),

      [=] __device__ (ptrdiff_t i){

          trip_count += 1;

      }
  );
  cudaDeviceSynchronize();


  long result = (long)trip_count;
  //printf("result=%ld\n", result);

  ASSERT_EQ(result, N);
}


CUDA_TEST(Nested, CudaExec_1threadblockexec){
  using namespace RAJA;
  using namespace RAJA::nested;

  constexpr long N = (long)1024*1024;

  // Loop Fusion
  using Pol = nested::Policy<
            CudaKernel<
              For<0, cuda_block_thread_exec<73>, Lambda<0>>
            >
        >;


  RAJA::ReduceSum<cuda_reduce<1024>, long> trip_count(0);

  nested::forall(
      Pol{},

      RAJA::make_tuple(RangeSegment(0,N)),

      [=] __device__ (ptrdiff_t i){

          trip_count += 1;

      }
  );
  cudaDeviceSynchronize();


  long result = (long)trip_count;
  //printf("result=%ld\n", result);

  ASSERT_EQ(result, N);
}

CUDA_TEST(Nested, CudaExec_2threadexec){
  using namespace RAJA;
  using namespace RAJA::nested;

  constexpr long N = (long)1024;

  // Loop Fusion
  using Pol = nested::Policy<
            CudaKernel<
              For<0, cuda_thread_exec,
                For<1, cuda_thread_exec, Lambda<0>>
              >
            >
        >;


  RAJA::ReduceSum<cuda_reduce<1024>, long> trip_count(0);

  nested::forall(
      Pol{},

      RAJA::make_tuple(RangeSegment(0,N), RangeSegment(0,N)),

      [=] __device__ (ptrdiff_t i, ptrdiff_t j){
          trip_count += 1;
      }
  );
  cudaDeviceSynchronize();

  long result = (long)trip_count;
  //printf("result=%ld\n", result);

  ASSERT_EQ(result, N*N);
}

CUDA_TEST(Nested, CudaExec_1thread1blockexec){
  using namespace RAJA;
  using namespace RAJA::nested;

  constexpr long N = (long)1024;

  // Loop Fusion
  using Pol = nested::Policy<
            CudaKernel<
              For<0, cuda_block_exec,
                For<1, cuda_thread_exec, Lambda<0>>
              >
            >
        >;


  RAJA::ReduceSum<cuda_reduce<1024>, long> trip_count(0);

  nested::forall(
      Pol{},

      RAJA::make_tuple(RangeSegment(0,N), RangeSegment(0,N)),

      [=] __device__ (ptrdiff_t i, ptrdiff_t j){

        if(i==j){
          trip_count += 1;
        }
      }
  );
  cudaDeviceSynchronize();

  long result = (long)trip_count;
  //printf("result=%ld\n", result);

  ASSERT_EQ(result, N);
}


CUDA_TEST(Nested, CudaExec_3threadexec){
  using namespace RAJA;
  using namespace RAJA::nested;

  constexpr long N = (long)256;

  // Loop Fusion
  using Pol = nested::Policy<
            CudaKernel<
              For<0, cuda_thread_exec,
                For<1, cuda_thread_exec,
                  For<2, cuda_thread_exec, Lambda<0>>
                >
              >
            >
        >;


  RAJA::ReduceSum<cuda_reduce<1024>, long> trip_count(0);

  nested::forall(
      Pol{},

      RAJA::make_tuple(RangeSegment(0,N), RangeSegment(0,N), RangeSegment(0,N)),

      [=] __device__ (ptrdiff_t i, ptrdiff_t j, ptrdiff_t k){

          trip_count += 1;

      }
  );
  cudaDeviceSynchronize();

  long result = (long)trip_count;
  //printf("result=%ld\n", result);

  ASSERT_EQ(result, N*N*N);
}


CUDA_TEST(Nested, CudaExec_tile1threadexec){
  using namespace RAJA;
  using namespace RAJA::nested;

  constexpr long N = (long)1024*1024;

  // Loop Fusion
  using Pol = nested::Policy<
            CudaKernel<
              nested::Tile<0, nested::tile_fixed<128>, seq_exec,
                For<0, cuda_thread_exec, Lambda<0>>
              >
            >
        >;


  RAJA::ReduceSum<cuda_reduce<1024>, long> trip_count(0);

  nested::forall(
      Pol{},

      RAJA::make_tuple(RangeSegment(0,N)),

      [=] __device__ (ptrdiff_t i){

          trip_count += 1;

      }
  );
  cudaDeviceSynchronize();


  long result = (long)trip_count;
  //printf("result=%ld\n", result);

  ASSERT_EQ(result, N);
}

#endif // CUDA
