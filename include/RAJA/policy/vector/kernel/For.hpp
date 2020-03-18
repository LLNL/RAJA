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

#ifndef RAJA_policy_vector_kernel_For_HPP
#define RAJA_policy_vector_kernel_For_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "RAJA/pattern/kernel/internal.hpp"
#include "RAJA/pattern/kernel/Lambda.hpp"
#include "RAJA/policy/vector/policy.hpp"

namespace RAJA
{

namespace internal
{




/*!
 * A generic RAJA::kernel forall_impl executor for statement::For
 *
 *
 */
template <camp::idx_t ArgumentId,
          typename VectorType,
          typename... EnclosedStmts,
          typename Types>
struct StatementExecutor<
    statement::For<ArgumentId, RAJA::vector_exec<VectorType>, EnclosedStmts...>,
    Types> {

  using stmt_list_t = StatementList<EnclosedStmts...>;


  template <typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {

    using DataT = camp::decay<Data>;


    auto begin = camp::get<ArgumentId>(data.segment_tuple).begin();
    auto end = camp::get<ArgumentId>(data.segment_tuple).end();
    auto distance = std::distance(begin, end);
    using diff_t = decltype(distance);


    diff_t distance_simd = distance - (distance%VectorType::s_num_elem);
    diff_t distance_remainder = distance - distance_simd;

    // compute the vector index type and new LoopTypes
    using value_type = camp::at_v<typename DataT::index_types_t, ArgumentId>;
    using vector_index_type = VectorIndex<value_type, VectorType>;
    using NewTypes = setSegmentType<Types, ArgumentId, vector_index_type>;


    // Streaming loop for complete vector widths
    camp::get<ArgumentId>(data.vector_sizes) = VectorType::s_num_elem;
    for (diff_t i = 0; i < distance_simd; i+=VectorType::s_num_elem) {

      data.template assign_offset<ArgumentId>(i);
      execute_statement_list<stmt_list_t, NewTypes>(data);

    }

    // Postamble for reamining elements
    if(distance_remainder > 0){
      camp::get<ArgumentId>(data.vector_sizes) = distance_remainder;

      data.template assign_offset<ArgumentId>(distance_simd);
      execute_statement_list<stmt_list_t, NewTypes>(data);
    }


  }
};


}  // namespace internal
}  // end namespace RAJA


#endif
