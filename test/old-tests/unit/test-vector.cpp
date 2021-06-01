//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for basic simd/simt vector operations
///

#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"
#include "RAJA_gtest.hpp"

#if 1

using VectorTestTypes = ::testing::Types<

  // Test automatically wrapped types, since the specific register
  // implementations are tested elsewhere
  RAJA::VectorRegister<int>,
  RAJA::VectorRegister<long>,
  RAJA::VectorRegister<float>,
  RAJA::VectorRegister<double>
  >;


#if 0
struct storage_policy{};

TEST(foobar, TestBlock)
{

  using block_t = RAJA::TensorBlock<RAJA::avx2_register, double,
      camp::idx_seq<0,1>,
      camp::idx_seq<16,16>,
      storage_policy>;


  block_t block;

  using matrix_t = block_t;

//  using matrix_t = RAJA::MatrixRegister<double, RAJA::ColMajorLayout>;
  using element_t = typename matrix_t::element_type;

  static const int N = 35;

  // Create a row-major data buffer
  element_t data1[N][N];
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      data1[i][j] = i*j*j;
    }
  }

  // Create an empty data bufffer
  element_t data2[N][N];

  //  Create views
  RAJA::View<element_t, RAJA::Layout<2, int>> view1(
      &data1[0][0], RAJA::make_permuted_layout<2, int>({{N, N}}, {{1,0}}));


  RAJA::View<element_t, RAJA::Layout<2, int>> view2(
      &data2[0][0], RAJA::make_permuted_layout<2, int>({{N, N}}, {{1,0}}));


  using Row = RAJA::RowIndex<int, matrix_t>;
  using Col = RAJA::ColIndex<int, matrix_t>;


  // Perform copy of view1 into view2
  view2(Row::all(), Col::all()) = 3 + 1*view1(Row::all(), Col::all()) - 3;


  // Check that data1==data2
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      ASSERT_SCALAR_EQ(data1[i][j], data2[i][j]);
    }
  }

  // Perform transpose view1 into view2 by switching col and row for view1
  view2(Row::all(), Col::all()) = view1(Col::all(), Row::all());

  // Check that data1==transpose(data2)
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      ASSERT_SCALAR_EQ(data1[i][j], data2[j][i]);
    }
  }

  // Perform transpose view1 into view2 by switching col and row for view2
  view2(Col::all(), Row::all()) = view1(Row::all(), Col::all());

  // Check that data1==transpose(data2)
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      ASSERT_SCALAR_EQ(data1[i][j], data2[j][i]);
    }
  }


}
#endif


template <typename Policy>
class VectorTest : public ::testing::Test
{
protected:

  VectorTest() = default;
  virtual ~VectorTest() = default;

  virtual void SetUp()
  {
  }

  virtual void TearDown()
  {
  }

};
TYPED_TEST_SUITE_P(VectorTest);


TYPED_TEST_P(VectorTest, GetSet)
{
  using vector_t = TypeParam;

  using element_t = typename vector_t::element_type;

  element_t A[vector_t::s_num_elem];
  for(camp::idx_t i = 0;i < vector_t::s_num_elem;++ i){
    A[i] = (element_t)(i*2);
  }

  // For Fixed vectors, only try with fixed length
  // For Stream vectors, try all lengths
  for(camp::idx_t N = 1; N <= vector_t::s_num_elem; ++ N){

    // load array A as vector
    vector_t vec;
    vec.load_packed_n(&A[0], N);

    // check get operations
    for(camp::idx_t i = 0;i < N;++ i){
      ASSERT_SCALAR_EQ(vec.get(i), (element_t)(i*2));
    }

    // check set operations
    for(camp::idx_t i = 0;i < N;++ i){
      vec.set((element_t)(i+1), i);
    }
    for(camp::idx_t i = 0;i < N;++ i){
      ASSERT_SCALAR_EQ(vec.get(i), (element_t)(i+1));
    }

  }
}

TYPED_TEST_P(VectorTest, MinMaxSumDot)
{
  using vector_t = TypeParam;

  using element_t = typename vector_t::element_type;

  element_t A[vector_t::s_num_elem];
  for(camp::idx_t i = 0;i < vector_t::s_num_elem;++ i){
    A[i] = (element_t)i;
  }

  // For Fixed vectors, only try with fixed length
  // For Stream vectors, try all lengths
  for(camp::idx_t N = 1; N <= vector_t::s_num_elem; ++ N){

    // load array A as vector
    vector_t vec;
    vec.load_packed_n(&A[0], N);

    // check min
    ASSERT_SCALAR_EQ(vec.min_n(N), (element_t)0);

    // check max
    ASSERT_SCALAR_EQ(vec.max_n(N), (element_t)(N-1));

    // compute expected values
    element_t ex_sum(0);
    element_t ex_dot(0);
    for(camp::idx_t i = 0;i < N;++ i){
      ex_sum += A[i];
      ex_dot += A[i]*A[i];
    }

    // check sum
    ASSERT_SCALAR_EQ(vec.sum(), ex_sum);

    // check dot
    ASSERT_SCALAR_EQ(vec.dot(vec), ex_dot);

  }
}


TYPED_TEST_P(VectorTest, FmaFms)
{
  using vector_t = TypeParam;

  using element_t = typename vector_t::element_type;

  element_t A[vector_t::s_num_elem];
  element_t B[vector_t::s_num_elem];
  element_t C[vector_t::s_num_elem];
  for(camp::idx_t i = 0;i < vector_t::s_num_elem;++ i){
    A[i] = (element_t)i;
    B[i] = (element_t)i*2;
    C[i] = (element_t)i*3;
  }

  // For Fixed vectors, only try with fixed length
  // For Stream vectors, try all lengths
  for(camp::idx_t N = 1; N <= vector_t::s_num_elem; ++ N){

    // load arrays as vectors
    vector_t vec_A;
    vec_A.load_packed_n(&A[0], N);

    vector_t vec_B;
    vec_B.load_packed_n(&B[0], N);

    vector_t vec_C;
    vec_C.load_packed_n(&C[0], N);


    // check FMA (A*B+C)

    vector_t fma = vec_A.multiply_add(vec_B, vec_C);
    for(camp::idx_t i = 0;i < N;++ i){
      ASSERT_SCALAR_EQ(fma.get(i), A[i]*B[i]+C[i]);
    }

    // check FMS (A*B-C)
    vector_t fms = vec_A.multiply_subtract(vec_B, vec_C);
    for(camp::idx_t i = 0;i < N;++ i){
      ASSERT_SCALAR_EQ(fms.get(i), A[i]*B[i]-C[i]);
    }

  }
}


TYPED_TEST_P(VectorTest, ForallVectorRef1d)
{
  using vector_t = TypeParam;

  using element_t = typename vector_t::element_type;


  size_t N = 10*vector_t::s_num_elem+1;
  // If we are not using fixed vectors, add some random number of elements
  // to the array to test some postamble code generation.
    //N += (size_t)(100*NO_OPT_RAND);


  element_t *A = new element_t[N];
  element_t *B = new element_t[N];
  element_t *C = new element_t[N];
  for(size_t i = 0;i < N; ++ i){
    A[i] = (element_t)(NO_OPT_RAND*1000.0);
    B[i] = (element_t)(NO_OPT_RAND*1000.0);
    C[i] = 0.0;
  }

  RAJA::View<element_t, RAJA::Layout<1>> X(A, N);
  RAJA::View<element_t, RAJA::Layout<1>> Y(B, N);
  RAJA::View<element_t, RAJA::Layout<1>> Z(C, N);


  using idx_t = RAJA::VectorIndex<int, vector_t>;

  auto all = idx_t::all();

  Z[all] = 3 + (X[all]*(5/Y[all])) + 9;

//  for(size_t i = 0;i < N; ++ i){
//    printf("%lf ", (double)C[i]);
//  }
//  printf("\n\n");

  for(size_t i = 0;i < N;i ++){
    ASSERT_SCALAR_EQ(element_t(3+(A[i]*(5/B[i]))+9), C[i]);
  }


  for(size_t i = 0;i < N; ++ i){
    C[i] = 0.0;
  }

  // evaluate on a subrange [N/2, N)
  auto some = idx_t::range(N/2, N);
  Z[some] = 3.+ (X[some]*(5/Y[some])) + 9;


  for(size_t i = 0;i < N/2;i ++){
    ASSERT_SCALAR_EQ(0, C[i]);
  }
  for(size_t i = N/2;i < N;i ++){
    ASSERT_SCALAR_EQ(element_t(3+(A[i]*(5/B[i]))+9), C[i]);
  }




  // evaluate on a subrange [0, N/2) using a forall statement
  for(size_t i = 0;i < N; ++ i){
    C[i] = 0.0;
  }
  RAJA::forall<RAJA::vector_exec<vector_t>>(RAJA::TypedRangeSegment<int>(0,N/2),
      [=](idx_t i){

     Z[i] = 3 + (X[i]*(5/Y[i])) + 9;
  });


  for(size_t i = 0;i < N/2;i ++){
    ASSERT_SCALAR_EQ(element_t(3+(A[i]*(5/B[i]))+9), C[i]);
  }
  for(size_t i = N/2;i < N;i ++){
    ASSERT_SCALAR_EQ(0, C[i]);
  }





  delete[] A;
  delete[] B;
  delete[] C;
}


TYPED_TEST_P(VectorTest, ForallVectorRef2d)
{
  using vector_t = TypeParam;
  using index_t = ptrdiff_t;

  using element_t = typename vector_t::element_type;


  index_t N = 3*vector_t::s_num_elem+1;
  index_t M = 4*vector_t::s_num_elem+1;
  // If we are not using fixed vectors, add some random number of elements
  // to the array to test some postamble code generation.
  N += (size_t)(10*NO_OPT_RAND);
  M += (size_t)(10*NO_OPT_RAND);

  element_t *A = new element_t[N*M];
  element_t *B = new element_t[N*M];
  element_t *C = new element_t[N*M];
  for(index_t i = 0;i < N*M; ++ i){
    A[i] = (element_t)(NO_OPT_RAND*1000.0);
    B[i] = (element_t)(NO_OPT_RAND*1000.0);
    C[i] = 0.0;
  }

  RAJA::View<element_t, RAJA::Layout<2>> X(A, N, M);
  RAJA::View<element_t, RAJA::Layout<2>> Y(B, N, M);
  RAJA::View<element_t, RAJA::Layout<2>> Z(C, N, M);

  using idx_t = RAJA::VectorIndex<index_t, vector_t>;
  auto all = idx_t::all();




  //
  // Test with kernel, using sequential policies and ::all()
  //
  for(index_t i = 0;i < N*M; ++ i){
    C[i] = 0.0;
  }
  using policy1_t =
      RAJA::KernelPolicy<
        RAJA::statement::For<0, RAJA::loop_exec,
            RAJA::statement::Lambda<0>
        >
      >;





  // Test with kernel, using sequential policies and ::all()
  RAJA::kernel<policy1_t>(
      RAJA::make_tuple(RAJA::TypedRangeSegment<index_t>(0, N)),

      [=](index_t i)
  {
    Z(i,all) = 3+(X(i,all)*(5/Y(i,all)))+9;
  });

  for(index_t i = 0;i < N*M;i ++){
    ASSERT_SCALAR_EQ(3+(A[i]*(5/B[i]))+9, C[i]);
  }


#if 1

  //
  // Test with kernel, using tensor_exec policy
  //


  for(index_t i = 0;i < N*M; ++ i){
    C[i] = 0.0;
  }

  using policy2_t =
      RAJA::KernelPolicy<
        RAJA::statement::For<0, RAJA::loop_exec,
          RAJA::statement::For<1, RAJA::vector_exec<vector_t>,
            RAJA::statement::Lambda<0>
          >
        >
      >;

  RAJA::kernel<policy2_t>(
      RAJA::make_tuple(RAJA::TypedRangeSegment<index_t>(0, N),
                       RAJA::TypedRangeSegment<index_t>(0, M)),

      [=](index_t i, idx_t j)
  {
    Z(i, j) = 3+(X(i, j)*(5/Y(i, j)))+9;
  });

  for(index_t i = 0;i < N*M;i ++){
    ASSERT_SCALAR_EQ(3+(A[i]*(5/B[i]))+9, C[i]);
  }




  //
  // Test with forall with vectors in i
  //
  for(index_t i = 0;i < N*M; ++ i){
    C[i] = 0.0;
  }
  RAJA::forall<RAJA::loop_exec>(RAJA::TypedRangeSegment<index_t>(0, M),
      [=](index_t j){


    Z(all,j) = 3+(X(all,j)*(5/Y(all,j)))+9;


  });

  for(index_t i = 0;i < N*M;i ++){
    ASSERT_SCALAR_EQ(3+(A[i]*(5/B[i]))+9, C[i]);
  }



  //
  // Test with forall with vectors in j
  //
  for(index_t i = 0;i < N*M; ++ i){
    C[i] = 0.0;
  }
  RAJA::forall<RAJA::loop_exec>(RAJA::TypedRangeSegment<index_t>(0, N),
      [=](index_t i){


    Z(i,all) = 3+(X(i,all)*(5/Y(i,all)))+9;


  });

  for(index_t i = 0;i < N*M;i ++){
    ASSERT_SCALAR_EQ(3+(A[i]*(5/B[i]))+9, C[i]);
  }


#endif


  delete[] A;
  delete[] B;
  delete[] C;
}


REGISTER_TYPED_TEST_SUITE_P(VectorTest,
    GetSet,
    MinMaxSumDot,
    FmaFms,
    ForallVectorRef1d,
    ForallVectorRef2d
    );

INSTANTIATE_TYPED_TEST_SUITE_P(SIMD, VectorTest, VectorTestTypes);

#endif

#if defined(RAJA_ENABLE_CUDA)


GPU_TEST(VectorTestCuda, CudaWarpVector)
{
  using namespace RAJA::statement;

  using element_t = double;
  size_t N = 32*5;

  element_t *A = nullptr;
  cudaErrchk(cudaMallocManaged(&A,
                    sizeof(element_t) * N,
                    cudaMemAttachGlobal));

  element_t *B = nullptr;

  cudaErrchk(cudaMallocManaged(&B,
                    sizeof(element_t) * N,
                    cudaMemAttachGlobal));


  element_t *C = nullptr;

  cudaErrchk(cudaMallocManaged(&C,
                    sizeof(element_t) * N,
                    cudaMemAttachGlobal));

  cudaErrchk(cudaDeviceSynchronize());


  for(size_t i = 0;i < N; ++ i){
    A[i] = (element_t)(NO_OPT_RAND*1000.0);
    B[i] = (element_t)(NO_OPT_RAND*1000.0);
    C[i] = 0.0;
  }

  RAJA::View<element_t, RAJA::Layout<1, int, 0>> X(A, N);
  RAJA::View<element_t, RAJA::Layout<1, int, 0>> Y(B, N);
  RAJA::View<element_t, RAJA::Layout<1, int, 0>> Z(C, N);


  cudaErrchk(cudaDeviceSynchronize());


  using vector_t = RAJA::VectorRegister<double, RAJA::cuda_warp_register>;
  using idx_t = RAJA::VectorIndex<int, vector_t>;
  auto all = idx_t::all();

  using pol_launch = RAJA::expt::LaunchPolicy<RAJA::expt::seq_launch_t, RAJA::expt::cuda_launch_t<true , 512> >;

  RAJA::expt::launch<pol_launch>(
      RAJA::expt::DEVICE,
      RAJA::expt::Resources(RAJA::expt::Teams(1),
                            RAJA::expt::Threads(32)),
      [=] RAJA_HOST_DEVICE (RAJA::expt::LaunchContext ctx)
  {


    Z[all] = 3 + (X[all]*(5/Y[all])) + 9;

  });


  cudaErrchk(cudaDeviceSynchronize());


  for(size_t i = 0;i < N;i ++){
    ASSERT_SCALAR_EQ(element_t(3+(A[i]*(5/B[i]))+9), C[i]);
  }




}



#endif // RAJA_ENABLE_CUDA

