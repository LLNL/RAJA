//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TESNOR_REGISTER_SegmentedSumOuter_HPP__
#define __TEST_TESNOR_REGISTER_SegmentedSumOuter_HPP__

#include<RAJA/RAJA.hpp>

template <typename REGISTER_TYPE>
void SegmentedSumOuterImpl()
{
  using register_type = REGISTER_TYPE;
  using element_type = typename register_type::element_type;
  using policy_type = typename register_type::register_policy;

  static constexpr camp::idx_t num_elem = register_type::s_num_elem;

  // Allocate

  std::vector<element_type> input0_vec(num_elem);
  element_type *input0_hptr = input0_vec.data();
  element_type *input0_dptr = tensor_malloc<policy_type, element_type>(num_elem);

  std::vector<element_type> output0_vec(num_elem);
  element_type *output0_hptr = output0_vec.data();
  element_type *output0_dptr = tensor_malloc<policy_type, element_type>(num_elem);


  // Initialize input data
  for(camp::idx_t i = 0;i < num_elem; ++ i){
    input0_hptr[i] = (element_type)(i+1); //+NO_OPT_RAND);
  }
  tensor_copy_to_device<policy_type>(input0_dptr, input0_vec);



  // run segmented dot products for all segments allowed by the vector
  for(int segbits = 0;(1<<segbits) <= num_elem;++ segbits){

    int num_segments = num_elem>>segbits;

    for(int output_segment = 0;output_segment < num_segments;++ output_segment){

      // Execute segmented broadcast
      tensor_do<policy_type>([=] RAJA_HOST_DEVICE (){

        register_type x;
        x.load_packed(input0_dptr);

        register_type y = x.segmented_sum_outer(segbits, output_segment);

        y.store_packed(output0_dptr);

      });

      // Move result to host
      tensor_copy_to_host<policy_type>(output0_vec, output0_dptr);


      // Check result

      // Compute expected values
      element_type expected[num_elem];
      for(camp::idx_t i = 0;i < num_elem; ++ i){
        expected[i] = 0;
      }

      int output_offset = output_segment * (1<<segbits);

      for(camp::idx_t i = 0;i < num_elem; ++ i){
        camp::idx_t output_i = output_offset + i%(1<<segbits);
        expected[output_i] += input0_hptr[i];
      }


      for(camp::idx_t i = 0;i < num_elem; ++ i){

        ASSERT_SCALAR_EQ(expected[i], output0_hptr[i]);
      }

    } // segment

  } // segbits


  // Cleanup
  tensor_free<policy_type>(input0_dptr);
  tensor_free<policy_type>(output0_dptr);
}



TYPED_TEST_P(TestTensorRegister, SegmentedSumOuter)
{
  SegmentedSumOuterImpl<TypeParam>();
}


#endif
