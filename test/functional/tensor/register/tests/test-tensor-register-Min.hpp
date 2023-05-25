//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TENSOR_REGISTER_Min_HPP__
#define __TEST_TENSOR_REGISTER_Min_HPP__

#include<RAJA/RAJA.hpp>

template <typename REGISTER_TYPE>
void MinImpl()
{
  using register_t = REGISTER_TYPE;
  using element_t = typename register_t::element_type;
  using policy_t = typename register_t::register_policy;

  static constexpr camp::idx_t num_elem = register_t::s_num_elem;

  // Allocate

  std::vector<element_t> input0_vec(num_elem);
  element_t *input0_hptr = input0_vec.data();
  element_t *input0_dptr = tensor_malloc<policy_t, element_t>(num_elem);

  std::vector<element_t> input1_vec(num_elem);
  element_t *input1_dptr = tensor_malloc<policy_t, element_t>(num_elem);

  std::vector<element_t> output0_vec(1);
  element_t *output0_dptr = tensor_malloc<policy_t, element_t>(1);

  std::vector<element_t> output1_vec(num_elem);
  element_t *output1_dptr = tensor_malloc<policy_t, element_t>(num_elem);


  // Initialize input data
  for(camp::idx_t i = 0;i < num_elem; ++ i){
   input0_hptr[i] = (element_t)(rand()*1000/RAND_MAX);
   input0_hptr[i] = (element_t)(rand()*1000/RAND_MAX);
  }

  tensor_copy_to_device<policy_t>(input0_dptr, input0_vec);
  tensor_copy_to_device<policy_t>(input1_dptr, input1_vec);


  //
  //  Check full-length operations
  //

  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    // load input vectors
    register_t x;
    x.load_packed(input0_dptr);

    register_t y;
    y.load_packed(input1_dptr);


    // compute reduction
    output0_dptr[0] = x.min();


    // compute element-wise
    register_t z = x.vmin(y);
    z.store_packed(output1_dptr);
  });

  tensor_copy_to_host<policy_t>(output0_vec, output0_dptr);
  tensor_copy_to_host<policy_t>(output1_vec, output1_dptr);



  // compute expected value for reduction
  element_t expected = input0_vec[0];
  for(camp::idx_t i = 1;i < num_elem;++i){
    expected = expected > input0_vec[i] ? input0_vec[i] : expected;
  }

  // check reduction
  ASSERT_SCALAR_EQ(expected, output0_vec[0]);


  // check element-wise operation
  for(camp::idx_t i = 0;i < num_elem;++i){
    ASSERT_SCALAR_EQ(std::min<element_t>(input0_vec[i], input1_vec[i]), output1_vec[i]);
  }


  //
  // check variable length operator
  //
  for(camp::idx_t N = 0;N <= num_elem;++ N){
    //
    //  Check full-length operations
    //

    tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

      register_t x;
      x.load_packed(input0_dptr);

      output0_dptr[0] = x.min_n(N);

    });

    tensor_copy_to_host<policy_t>(output0_vec, output0_dptr);


    // compute expected value for reduction
    element_t expected = RAJA::operators::limits<element_t>::max();
    for(camp::idx_t i = 0;i < N;++i){
      expected = expected > input0_vec[i] ? input0_vec[i] : expected;
    }

    // check reduction
    ASSERT_SCALAR_EQ(expected, output0_vec[0]);

  }

  // Cleanup
  tensor_free<policy_t>(input0_dptr);
  tensor_free<policy_t>(input1_dptr);
  tensor_free<policy_t>(output0_dptr);
  tensor_free<policy_t>(output1_dptr);
}



TYPED_TEST_P(TestTensorRegister, Min)
{
  MinImpl<TypeParam>();
}


#endif
