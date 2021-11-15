//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TESNOR_REGISTER_SegmentedDotProduct_HPP__
#define __TEST_TESNOR_REGISTER_SegmentedDotProduct_HPP__

#include<RAJA/RAJA.hpp>

template <typename REGISTER_TYPE>
void SegmentedDotProductImpl()
{
  using register_t = REGISTER_TYPE;
  using element_t = typename register_t::element_type;
  using policy_t = typename register_t::register_policy;

  static constexpr size_t num_elem = register_t::s_num_elem;

  element_t A[num_elem], B[num_elem], R[num_elem];
  register_t x, y;

  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = (element_t)(NO_OPT_RAND*1000.0);
    B[i] = (element_t)(NO_OPT_RAND*1000.0);
    x.set(A[i], i);
    y.set(B[i], i);
  }

//  printf("x: %s", x.to_string().c_str());
//  printf("y: %s", y.to_string().c_str());


  // run segmented dot products for all segments allowed by the vector
  for(int segbits = 0;(1<<segbits) <= num_elem;++ segbits){

    int num_output_segments = 1<<segbits;

    for(int output_segment = 0;output_segment < num_output_segments;++output_segment){

      int offset = output_segment * num_elem/(1<<segbits);

      register_t dp = x.segmented_dot(segbits, output_segment, y);


      // Compute expected values
      for(size_t i = 0;i < num_elem; ++ i){
        R[i] = 0;
      }
      for(size_t i = 0;i < num_elem; ++ i){
        R[(i>>segbits) + offset] += A[i]*B[i];
      }

      //printf("xdoty: segbits=%d, oseg=%d, %s", segbits, output_segment, dp.to_string().c_str());

      for(size_t i = 0;i < num_elem; ++ i){
        //printf("i=%d, R=%lf, dp=%lf\n", (int)i, (double)R[i], (double)dp.get(i));
        ASSERT_SCALAR_EQ(R[i], dp.get(i));
      }

    } // output_segment

  } // segbits

}



TYPED_TEST_P(TestTensorRegister, SegmentedDotProduct)
{
  SegmentedDotProductImpl<TypeParam>();
}


#endif
