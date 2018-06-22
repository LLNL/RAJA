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


void myfunc(){

  int N = 33;

  std::vector<double> a_data(N*2);
  std::vector<double> b_data(N);
  std::vector<double> c_data(N);

  View<double, Layout<1>> a(&a_data[0], N);
  View<double, Layout<1>> b(&b_data[0], N);
  View<double, Layout<1>> c(&c_data[0], N);

  for(int i = 0;i < N; ++ i){
    a_data[i] = i;
    b_data[i] = i*i;
    c_data[i] = i*i*i;
  }

  using VecType = vec::Vector<double, 2, 1>;

  using policy = vec_exec<VecType>;
  //using policy = seq_exec;

  auto va = make_vector_view<VecType, 0>(a);
  auto vb = make_vector_view<VecType, 0>(b);
  auto vc = make_vector_view<VecType, 0>(c);

  printf("START\n");

  forall<policy>(RangeSegment(0, N),
      [&] (auto i){

        //vec::StridedVector<double, 2, 1> vs{&a_data[i.value], 2};

        //vs = a(i) + b(i)*c(i);

        //va(i) = (vb(i))*((VecType)vc(i));
        va(i) = vb(i)*vc(i);

      });

  printf("DONE\n");

  for(int i = 0;i < N;++ i){
    printf("a[%d] = %lf\n", i, a_data[i]);
    //ASSERT_EQ(a_data[i], (double)(i + i*i*i*i*i));
  }

 }
TEST(Vec, simple){
myfunc();

}
