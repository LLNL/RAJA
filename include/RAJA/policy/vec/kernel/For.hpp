/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for statement wrappers and executors.
 *
 ******************************************************************************
 */


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


#ifndef RAJA_policy_vec_kernel_For_HPP
#define RAJA_policy_vec_kernel_For_HPP


#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include <RAJA/policy/vec/policy.hpp>

namespace RAJA
{
namespace internal
{


template <camp::idx_t ArgumentId,
          typename VecType,
          typename... EnclosedStmts>
struct StatementExecutor<statement::
                             For<ArgumentId, vec_exec<VecType>, EnclosedStmts...>> {


  template <typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {
    using data_t = camp::decay<Data>;

    // compute vectorized and scalar trip counts
    static constexpr size_t element_width = VecType::num_total_elements;
 
    auto len = segment_length<ArgumentId>(data);
    using len_t = decltype(len);

    len_t vec_remainder = len % element_width;
    len_t vec_distance = len - vec_remainder;

    // Argument's original index type
    using IdxType = camp::at_v<typename data_t::argument_types, ArgumentId>;

    // Index to use for vector loop
    using VecIdx = RAJA::vec::VectorIndex<IdxType, VecType>;

    // Coerce LoopData into using the VecIdx type for ArgumentID
    auto &vec_data = overrideArgumentType<data_t, ArgumentId, VecIdx>(data);
    using VecData = camp::decay<decltype(vec_data)>;

    // Create a vectorized wrapper
    ForWrapper<ArgumentId, VecData, EnclosedStmts...> for_vec_wrapper(vec_data);




    // Index to use for scalar postamble loop
    using scalar_type = typename VecType::scalar_type;
    using ScalarType = RAJA::vec::Vector<scalar_type, 1, 1>;
    using ScalarIdx = RAJA::vec::VectorIndex<IdxType, ScalarType>;

    // Coerce LoopData into using the ScalarIdx type for ArgumentID
    auto &scalar_data = overrideArgumentType<data_t, ArgumentId, ScalarIdx>(data);
    using ScalarData = camp::decay<decltype(scalar_data)>;

    // Create a scalar wrapper
    ForWrapper<ArgumentId, ScalarData, EnclosedStmts...> for_scalar_wrapper(scalar_data);


    


    // Execute the vectorized loop portion
    for(len_t i = 0;i < vec_distance;i += element_width){
      for_vec_wrapper(i);
    }

    // Execute the scalar postamble loop portion
    for(len_t i = vec_distance;i < len;++ i){
      for_scalar_wrapper(i);
    }

  }
};


}  // namespace internal
}  // end namespace RAJA


#endif /* RAJA_pattern_nested_HPP */
