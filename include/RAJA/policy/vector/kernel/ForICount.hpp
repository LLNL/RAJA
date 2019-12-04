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
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_vector_kernel_ForICount_HPP
#define RAJA_policy_vector_kernel_ForICount_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "RAJA/pattern/kernel/internal.hpp"
#include "RAJA/pattern/kernel/Lambda.hpp"
#include "RAJA/policy/vector/policy.hpp"
#include "RAJA/policy/vector/kernel/For.hpp"

namespace RAJA
{

namespace internal
{



/*!
 * A vector_exec RAJA::kernel forall_impl executor for statement::For
 *
 *
 */

template <camp::idx_t ArgumentId, typename ParamId,
          typename... EnclosedStmts>
struct StatementExecutor<
    statement::ForICount<ArgumentId, ParamId, RAJA::vector_exec,
                         EnclosedStmts...>> {

  template <typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {

    // Create a wrapper, just in case forall_impl needs to thread_privatize
    ForICountWrapper<ArgumentId, Data, EnclosedStmts...> for_wrapper(data);


    auto &begin = camp::get<ArgumentId>(data.segment_tuple).begin();
    auto end = camp::get<ArgumentId>(data.segment_tuple).end();
    auto distance = std::distance(begin, end);
    using diff_t = decltype(distance);

    using Iterator = decltype(end);
    using vector_type = typename Iterator::vector_type;

    diff_t distance_simd = distance - (distance%vector_type::s_num_elem);
    diff_t distance_remainder = distance - distance_simd;

    // Streaming loop for complete vector widths
    begin.set_vector_length(vector_type::s_num_elem);
    for (diff_t i = 0; i < distance_simd; i+=vector_type::s_num_elem) {
      for_wrapper(i);
    }

    // Postamble for reamining elements
    if(distance_remainder > 0){
      begin.set_vector_length(distance_remainder);
      for_wrapper(distance_simd);
    }


  }
};


}  // namespace internal
}  // end namespace RAJA


#endif 
