//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
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

///
/// Source file containing tests for explicit vector policies
///

#include <RAJA/RAJA.hpp>
#include "RAJA_gtest.hpp"


using namespace RAJA;
using namespace RAJA::statement;

TEST(Vec, simple_1d_strided){

  int N = 33;

  std::vector<double> a_data(N+1);
  std::vector<double> b_data(N);
  std::vector<double> c_data(N);

  View<double, Layout<1>> a(&a_data[0], N+1);
  View<double, Layout<1>> b(&b_data[0], N);
  View<double, Layout<1>> c(&c_data[0], N);

  for(int i = 0;i < N+1; ++ i){
    a_data[i] = i;
  } 
  for(int i = 0;i < N; ++ i){
    b_data[i] = i*i;
    c_data[i] = i*i*i;
  }

  using VecType = vec::Vector<double, 2, 1>;

  using policy = vec_exec<VecType>;

  // create strided VectorViewWrappers 
  // (none of the Layouts have stride1_dim set)
  auto va = make_vector_view(a);
  auto vb = make_vector_view(b);
  auto vc = make_vector_view(c);

  forall<policy>(RangeSegment(0, N),
      [=] (auto i){

        va(i) += vb(i)*vc(i);

      });
  

  // check results
  for(int i = 0;i < N;++ i){
    ASSERT_EQ(a_data[i], (double)(i + i*i*i*i*i));
  }

  // make sure we didn't run off the end
  ASSERT_EQ(a_data[N], N);


}



TEST(Vec, simple_1d_packed){

  int N = 33;

  std::vector<double> a_data(N+1);
  std::vector<double> b_data(N);
  std::vector<double> c_data(N);

  View<double, Layout<1>> a(&a_data[0], N+1);
  View<double, Layout<1>> b(&b_data[0], N);
  View<double, Layout<1>> c(&c_data[0], N);

  for(int i = 0;i < N+1; ++ i){
    a_data[i] = i;
  } 
  for(int i = 0;i < N; ++ i){
    b_data[i] = i*i;
    c_data[i] = i*i*i;
  }

  using VecType = vec::Vector<double, 2, 1>;

  using policy = vec_exec<VecType>;

  // using pointers should enfore stride1
  auto va = make_vector_view(a);
  auto vb = make_vector_view(b);
  auto vc = make_vector_view(c);
  
  forall<policy>(RangeSegment(0, N),
      [=] (auto i){

        va(i) += vb(i)*vc(i);

      });
  

  // check results
  for(int i = 0;i < N;++ i){
    ASSERT_EQ(a_data[i], (double)(i + i*i*i*i*i));
  }

  // make sure we didn't run off the end
  ASSERT_EQ(a_data[N], N);


}
TEST(Vec, simple_reduce_sum){

  int N = 137;

  std::vector<double> a_data(N);


  View<double, Layout<1>> a(&a_data[0], N);

  for(int i = 0;i < N; ++ i){
    a_data[i] = 1;
  }

  using VecType = vec::Vector<double, 2, 1>;

  using policy = vec_exec<VecType>;

  // using pointers should enfore stride1
  auto va = make_vector_view(a);

  ReduceSum<vec_reduce, VecType> sum(0.0);
  ReduceSum<seq_reduce, int> trip_count(0);

  forall<policy>(RangeSegment(0, N),
      [=] (auto i){

        sum += va(i);
        trip_count += 1;

      });


  ASSERT_EQ((double)sum, 137.0);
  ASSERT_EQ((int)trip_count, 69);


}

TEST(Vec, simple_kernel_reduce_sum){

  int N = 137;

  std::vector<double> a_data(N);


  View<double, Layout<1>> a(&a_data[0], N);

  for(int i = 0;i < N; ++ i){
    a_data[i] = 1;
  }

  using VecType = vec::Vector<double, 2, 1>;

  using policy = RAJA::KernelPolicy<
    RAJA::statement::For<0, vec_exec<VecType>,
      RAJA::statement::Lambda<0>
    >
  >;

  // using pointers should enfore stride1
  auto va = make_vector_view(a);

  ReduceSum<vec_reduce, VecType> sum(0.0);
  ReduceSum<seq_reduce, int> trip_count(0);

  RAJA::kernel<policy>(
      RAJA::make_tuple(RangeSegment(0, N)),

      [=] (auto i){

        sum += va(i);
        trip_count += 1;

      });


  ASSERT_EQ((double)sum, 137.0);
  ASSERT_EQ((int)trip_count, 69);


}
